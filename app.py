"""Multimodal Deep Learning for Cloud Removal in Satellite Imagery.
Scientific Research Interface & Interactive Dashboard
ISRO Remote-Sensing & Deep Learning Research Architecture
"""

from collections import Counter
import io
import json
import os
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets.alignment import check_geospatial_alignment, validate_array_alignment
from datasets.cloud_masks import compute_cloud_fraction, detect_s2_clouds
from datasets.dataset_adapter import SEN12MSCRTSDataset
from datasets.preprocessing import (
    make_rgb_preview,
    make_sar_preview,
    normalize_optical,
    normalize_sar,
    read_geotiff_multimodal,
)
from evaluation.baselines import passthrough_baseline, spatial_inpaint_baseline
from evaluation.metrics import calculate_psnr, calculate_ssim, evaluate_sample_metrics
from inference.geotiff_io import read_satellite_image, write_satellite_image
from inference.predict import CloudRemovalPredictor
from training.losses import composite_cloud_free
from backend.core.config import settings
from visualization.plots import (
    plot_benchmark_bars,
    plot_reconstruction_results,
    plot_training_curves,
)

# Page configuration
st.set_page_config(
    page_title="Multimodal Satellite Cloud Removal",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for the research dashboard and workflow pages
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(15, 118, 110, 0.08), transparent 36%),
            #f5f7f8;
    }
    [data-testid="stHeader"] {
        background: rgba(245, 247, 248, 0.82);
    }
    [data-testid="stSidebar"] {
        background: #102b2a;
    }
    [data-testid="stSidebar"] * {
        color: #e7f1ee;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: #e7f1ee;
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1440px;
        padding-top: 2.2rem;
    }
    .main-header {
        font-size: clamp(1.8rem, 3.5vw, 2.65rem);
        font-weight: 750;
        letter-spacing: -0.04em;
        line-height: 1.12;
        color: #102b2a;
        margin: 0 0 0.45rem;
    }
    .sub-header {
        font-size: 1rem;
        line-height: 1.6;
        color: #5c6b69;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        height: 100%;
        min-height: 128px;
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid #e1e9e7;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 8px 24px rgba(16, 43, 42, 0.045);
        margin-bottom: 0.5rem;
    }
    .metric-title {
        font-size: 0.76rem;
        color: #647572;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }
    .metric-val {
        font-size: 1.35rem;
        font-weight: 750;
        letter-spacing: -0.025em;
        color: #102b2a;
        margin: 0.38rem 0 0.2rem;
    }
    .dashboard-panel {
        height: 100%;
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid #e1e9e7;
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: 0 8px 24px rgba(16, 43, 42, 0.04);
    }
    .dashboard-eyebrow {
        color: #0f766e;
        font-size: 0.75rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.65rem;
    }
    .dashboard-note {
        border-left: 3px solid #0f766e;
        border-radius: 0 10px 10px 0;
        background: #eaf3f1;
        color: #36504d;
        padding: 13px 16px;
        margin: 0.6rem 0 1rem;
        line-height: 1.55;
    }
    div.stButton > button[kind="secondary"] {
        border-color: #d5e2df;
        color: #124e4a;
        border-radius: 10px;
    }
    div.stButton > button[kind="primary"] {
        background: #0f766e;
        border-color: #0f766e;
        border-radius: 10px;
    }
    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1.2rem 1rem;
        }
        .metric-card {
            min-height: 110px;
            padding: 15px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("🛰️ System Navigation")
st.sidebar.caption("ISRO Remote-Sensing & DL Research")

section = st.sidebar.radio(
    "Modules",
    [
        "Dashboard",
        "Data Explorer",
        "Cloud Removal (Inference)",
        "Model Evaluation & Benchmarks",
        "Training Monitor",
        "Research Reports & Specs",
    ],
    key="section",
)

# Check device and checkpoints
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = Path("outputs/checkpoints")
available_checkpoints = list(checkpoint_dir.glob("*.pt")) if checkpoint_dir.is_dir() else []

# Data root
data_root = Path("data/raw/SEN12MSCRTS/extracted")
has_real_data = data_root.is_dir() and len(list(data_root.glob("**/*.tif"))) > 0

st.sidebar.markdown("---")
st.sidebar.subheader("Hardware & Runtime")
st.sidebar.info(
    f"**Device:** `{device}`\n\n"
    f"**PyTorch:** `{torch.__version__}`\n\n"
    f"**Checkpoints Found:** `{len(available_checkpoints)}`\n\n"
    f"**Dataset Status:** `{'SEN12MS-CR-TS Online' if has_real_data else 'Synthetic Mode'}`"
)


# ==============================================================================
# TAB 1: DASHBOARD
# ==============================================================================
if section == "Dashboard":
    inspection_path = Path("outputs/reports/dataset_inspection.json")
    inspection = json.loads(inspection_path.read_text(encoding="utf-8")) if inspection_path.is_file() else {}
    history_files = list(Path("outputs/reports").glob("training_history_*.json"))
    latest_history_path = max(history_files, key=lambda path: path.stat().st_mtime) if history_files else None
    latest_history = (
        json.loads(latest_history_path.read_text(encoding="utf-8"))
        if latest_history_path is not None
        else {}
    )
    comparison_path = Path("outputs/reports/model_comparison.csv")
    configured_checkpoint = settings.checkpoint_path
    configured_checkpoint_exists = configured_checkpoint.is_file()
    latest_epoch = latest_history.get("epoch", [])
    latest_train_loss = latest_history.get("loss_train", [])
    latest_val_psnr = latest_history.get("val_psnr", [])
    latest_val_ssim = latest_history.get("val_ssim", [])
    epoch_count = len(latest_epoch)

    st.markdown('<div class="dashboard-eyebrow">Earth observation · Research workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">Multimodal satellite cloud removal</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Explore SEN12MS-CR-TS observations, inspect experiment evidence, and reconstruct cloud-obscured optical imagery using Sentinel-1 SAR as complementary input.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        dataset_status = "Dataset available" if has_real_data else "Dataset not found"
        dataset_detail = (
            f"{inspection.get('patch_count', '—')} inspected patches"
            if inspection
            else "Inspection report not available"
        )
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-title">Dataset · SEN12MS-CR-TS</div>
            <div class="metric-val">{dataset_status}</div>
            <span style="font-size:0.84rem; color:#647572">{dataset_detail}</span>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        model_status = "Checkpoint ready" if configured_checkpoint_exists else "Model unavailable"
        model_detail = (
            configured_checkpoint.name
            if configured_checkpoint_exists
            else f"Expected: {configured_checkpoint.name}"
        )
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-title">Inference · API checkpoint</div>
            <div class="metric-val">{model_status}</div>
            <span style="font-size:0.84rem; color:#647572">{model_detail}</span>
            </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        training_status = f"{epoch_count} epochs logged" if epoch_count else "No run recorded"
        training_detail = latest_history_path.stem if latest_history_path else "Training history unavailable"
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-title">Latest training run</div>
            <div class="metric-val">{training_status}</div>
            <span style="font-size:0.84rem; color:#647572">{training_detail}</span>
            </div>""",
            unsafe_allow_html=True,
        )
    with col4:
        evaluation_status = "Results available" if comparison_path.is_file() else "No comparison report"
        evaluation_detail = comparison_path.name if comparison_path.is_file() else "No saved evaluation CSV"
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-title">Model evaluation</div>
            <div class="metric-val">{evaluation_status}</div>
            <span style="font-size:0.84rem; color:#647572">{evaluation_detail}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("")
    workflow_col, summary_col = st.columns([1.1, 1])
    with workflow_col:
        st.markdown("### Research workflow")
        st.caption("Choose a workspace to continue.")
        workflow_buttons = [
            ("Explore dataset", "Data Explorer"),
            ("Run cloud removal", "Cloud Removal (Inference)"),
            ("Review evaluation", "Model Evaluation & Benchmarks"),
            ("Open training monitor", "Training Monitor"),
        ]
        for row_start in range(0, len(workflow_buttons), 2):
            button_cols = st.columns(2)
            for button_col, (label, target) in zip(button_cols, workflow_buttons[row_start:row_start + 2]):
                button_col.button(
                    label,
                    key=f"dashboard_{target}",
                    use_container_width=True,
                    on_click=lambda destination=target: st.session_state.update({"section": destination}),
                )

        st.markdown(
            '<div class="dashboard-note"><strong>Scientific context.</strong> SAR backscatter and optical reflectance measure different physical properties. Treat radar as complementary evidence, and assess reconstruction against a valid paired clear reference where available.</div>',
            unsafe_allow_html=True,
        )
    with summary_col:
        st.markdown("### Latest experiment summary")
        if epoch_count:
            summary_cols = st.columns(3)
            summary_cols[0].metric("Last epoch", str(latest_epoch[-1]))
            if latest_train_loss:
                summary_cols[1].metric("Train loss", f"{latest_train_loss[-1]:.4f}")
            else:
                summary_cols[1].metric("Train loss", "Not recorded")
            summary_cols[2].metric(
                "Val. PSNR",
                f"{latest_val_psnr[-1]:.2f} dB" if latest_val_psnr else "Not recorded",
            )
            if latest_val_ssim:
                st.caption(f"Validation SSIM: {latest_val_ssim[-1]:.4f} · Values shown from the latest saved training history.")
            else:
                st.caption("Validation SSIM was not recorded in this training history.")
        else:
            st.info("No training history is available yet.")

        if comparison_path.is_file():
            comparison_df = pd.read_csv(comparison_path)
            st.caption(f"Saved comparison includes {len(comparison_df)} model result rows.")
        else:
            st.info("No saved held-out evaluation summary is available. Evaluation metrics are not inferred or fabricated.")

    st.markdown("### System overview")
    overview_cols = st.columns(3)
    overview_items = [
        ("01 · Observations", "Cloudy Sentinel-2 optical bands, corresponding Sentinel-1 VV/VH observations, and an optional cloud mask."),
        ("02 · Reconstruction", "A configurable multimodal U-Net predicts optical reflectance; it does not make SAR a substitute for hidden optical detail."),
        ("03 · Evidence", "Use matched references and the same held-out samples for quantitative comparisons. Preview-only inputs cannot establish scientific performance."),
    ]
    for overview_col, (title, description) in zip(overview_cols, overview_items):
        overview_col.markdown(
            f'<div class="dashboard-panel"><div class="dashboard-eyebrow">{title}</div><div style="color:#526461; line-height:1.6">{description}</div></div>',
            unsafe_allow_html=True,
        )


# ==============================================================================
# TAB 2: DATA EXPLORER
# ==============================================================================
elif section == "Data Explorer":
    st.markdown('<div class="main-header">Sentinel-1 & Sentinel-2 Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspect genuine SEN12MS-CR-TS multispectral bands, SAR polarizations, and coregistration metadata.</div>', unsafe_allow_html=True)

    split = st.selectbox("Select Split Partition", ["train", "val", "test"])

    try:
        ds = SEN12MSCRTSDataset(
            root_dir=data_root,
            split=split,
            optical_bands=13,
            sar_bands=2,
            use_mask=True,
            synthetic_fallback=True,
        )
        total_p = len(ds)
        st.success(f"Loaded **{total_p}** multimodal sample pairs for partition `{split}` (Synthetic: `{ds.is_synthetic}`).")

        sample_idx = st.slider("Select Sample Index", 0, max(total_p - 1, 0), 0)
        item = ds[sample_idx]

        opt_tensor = item["optical"].numpy()
        sar_tensor = item["sar"].numpy()
        mask_tensor = item["cloud_mask"].numpy()
        tgt_tensor = item["target"].numpy()
        meta = item["metadata"]

        # Convert to HWC
        opt_hwc = np.moveaxis(opt_tensor, 0, -1)
        sar_hwc = np.moveaxis(sar_tensor, 0, -1)
        mask_hwc = np.moveaxis(mask_tensor, 0, -1)
        tgt_hwc = np.moveaxis(tgt_tensor, 0, -1)

        # Metadata cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Geographic ROI", meta.get("roi", "N/A"))
        c2.metric("Patch ID", str(meta.get("patch_id", "N/A")))
        c3.metric("Cloudy Date", meta.get("cloudy_date", "N/A"))
        c4.metric("Cloud Fraction", f"{meta.get('cloud_fraction', 0.0) * 100:.1f}%")

        # Visualization controls
        st.markdown("#### Multispectral Band Visualizations")
        band_mode = st.radio(
            "Spectral Band Mapping:",
            [
                "True Color RGB (B4 - Red, B3 - Green, B2 - Blue)",
                "False Color Vegetation/Urban (B8 - NIR, B4 - Red, B3 - Green)",
                "Agriculture / SWIR (B11 - SWIR1, B8 - NIR, B2 - Blue)",
                "Cirrus & Atmospheric Water Vapour (B10 - Cirrus, B9 - Vapour, B1 - Aerosol)",
            ],
            horizontal=True,
        )

        band_map = (3, 2, 1)
        if "False Color" in band_mode:
            band_map = (7, 3, 2)
        elif "Agriculture" in band_mode:
            band_map = (11, 7, 1)
        elif "Cirrus" in band_mode:
            band_map = (10, 9, 0)

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.image(make_rgb_preview(opt_hwc, bands=band_map), caption=f"Cloudy Optical ({band_mode.split(' ')[0]})", use_container_width=True)
        with col_b:
            st.image(make_sar_preview(sar_hwc), caption="Sentinel-1 SAR (Dual-Pol Composite)", use_container_width=True)
        with col_c:
            st.image(mask_hwc[..., 0], caption=f"Detected Cloud Mask ({compute_cloud_fraction(mask_hwc)*100:.1f}%)", clamp=True, use_container_width=True)
        with col_d:
            st.image(make_rgb_preview(tgt_hwc, bands=band_map), caption="Clear Reference Optical (Target)", use_container_width=True)

        # Reflectance Histograms
        st.markdown("#### Spectral Reflectance Distribution")
        fig_hist, ax_h = plt.subplots(figsize=(8, 2.5), dpi=120)
        ax_h.hist(opt_hwc[..., 1].ravel(), bins=50, alpha=0.6, color="blue", label="B2 (Blue)")
        ax_h.hist(opt_hwc[..., 2].ravel(), bins=50, alpha=0.6, color="green", label="B3 (Green)")
        ax_h.hist(opt_hwc[..., 3].ravel(), bins=50, alpha=0.6, color="red", label="B4 (Red)")
        if opt_hwc.shape[-1] >= 8:
            ax_h.hist(opt_hwc[..., 7].ravel(), bins=50, alpha=0.4, color="purple", label="B8 (NIR)")
        ax_h.set_title("Surface Reflectance Histogram [0, 1]")
        ax_h.set_xlabel("Reflectance")
        ax_h.set_ylabel("Pixel Frequency")
        ax_h.legend(fontsize=8)
        st.pyplot(fig_hist)

    except Exception as e:
        st.error(f"Error exploring dataset: {e}")


# ==============================================================================
# TAB 3: CLOUD REMOVAL (INFERENCE)
# ==============================================================================
elif section == "Cloud Removal (Inference)":
    st.markdown('<div class="main-header">Satellite Cloud Removal & Reconstruction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Execute deep multimodal reconstruction on cloud-obscured Sentinel-2 optical imagery.</div>', unsafe_allow_html=True)

    input_source = st.radio("Input Source:", ["Select Sample from Real Dataset", "Upload Custom Satellite Files (GeoTIFF / PNG)"], horizontal=True)

    cloudy_opt_data = None
    sar_data = None
    mask_data = None
    reference_data = None
    input_profile = None

    if input_source == "Select Sample from Real Dataset":
        try:
            ds = SEN12MSCRTSDataset(data_root, split="test", synthetic_fallback=True)
            sample_id = st.slider("Test Sample Index", 0, len(ds) - 1, 0)
            item = ds[sample_id]
            cloudy_opt_data = np.moveaxis(item["optical"].numpy(), 0, -1)
            sar_data = np.moveaxis(item["sar"].numpy(), 0, -1)
            mask_data = np.moveaxis(item["cloud_mask"].numpy(), 0, -1)
            reference_data = np.moveaxis(item["target"].numpy(), 0, -1)
            st.info(f"Loaded Sample from ROI `{item['metadata']['roi']}`, Patch `{item['metadata']['patch_id']}` (Cloudy Date: `{item['metadata']['cloudy_date']}`).")
        except Exception as e:
            st.error(f"Error loading sample: {e}")

    else:
        st.markdown("##### Upload Satellite Files")
        st.caption("Upload GeoTIFF or PNG files. Values must be normalized or in standard digital numbers.")
        u1, u2 = st.columns(2)
        with u1:
            up_opt = st.file_uploader("1. Cloudy Optical Image (Sentinel-2)", type=["tif", "tiff", "png", "jpg"], key="up_opt")
        with u2:
            up_sar = st.file_uploader("2. Corresponding SAR Image (Sentinel-1)", type=["tif", "tiff", "png", "jpg"], key="up_sar")

        u3, u4 = st.columns(2)
        with u3:
            up_mask = st.file_uploader("3. Optional Binary Cloud Mask (1=Cloud, 0=Clear)", type=["tif", "tiff", "png"], key="up_mask")
        with u4:
            up_ref = st.file_uploader("4. Optional Clear Reference Image", type=["tif", "tiff", "png", "jpg"], key="up_ref")

        if up_opt and up_sar:
            try:
                arr_opt, input_profile = read_satellite_image(up_opt.getvalue(), up_opt.name)
                arr_sar, _ = read_satellite_image(up_sar.getvalue(), up_sar.name)
                cloudy_opt_data = normalize_optical(arr_opt)
                sar_data = normalize_sar(arr_sar)

                if up_mask:
                    arr_mask, _ = read_satellite_image(up_mask.getvalue(), up_mask.name)
                    mask_data = (arr_mask > 0.5).astype(np.float32)
                    if mask_data.ndim == 2:
                        mask_data = mask_data[..., np.newaxis]

                if up_ref:
                    arr_ref, _ = read_satellite_image(up_ref.getvalue(), up_ref.name)
                    reference_data = normalize_optical(arr_ref)

                st.success("Successfully loaded and normalized uploaded satellite observations.")
            except Exception as e:
                st.error(f"Error parsing uploaded files: {e}")

    st.markdown("---")
    st.markdown("##### Inference Configuration")
    c_inf1, c_inf2, c_inf3 = st.columns(3)

    with c_inf1:
        ckpt_options = [p.name for p in available_checkpoints] or ["multimodal_unet.pt (untrained demo)"]
        selected_ckpt = st.selectbox("Select Model Checkpoint:", ckpt_options)

    with c_inf2:
        use_comp = st.checkbox("Enable Mask-Aware Compositing", value=True, help="Blends model reconstruction on cloud pixels with original high-confidence clear pixels.")

    with c_inf3:
        run_btn = st.button("🚀 Reconstruct Cloud-Free Image", type="primary", use_container_width=True)

    if run_btn:
        if cloudy_opt_data is None or sar_data is None:
            st.error("Please supply both Cloudy Optical and Sentinel-1 SAR inputs.")
        else:
            ckpt_path = checkpoint_dir / selected_ckpt if (checkpoint_dir / selected_ckpt).is_file() else None

            with st.spinner("Executing multimodal deep learning reconstruction..."):
                try:
                    if ckpt_path and ckpt_path.is_file():
                        predictor = CloudRemovalPredictor(ckpt_path)
                        pred_out = predictor.predict(
                            cloudy_optical=cloudy_opt_data,
                            sar=sar_data,
                            cloud_mask=mask_data,
                            reference=reference_data,
                            use_compositing=use_comp,
                        )
                    else:
                        # Fallback for on-the-fly demo if checkpoint not yet trained
                        st.warning("Running model forward pass with initial weights...")
                        model = MultimodalUNet(optical_channels=cloudy_opt_data.shape[-1], sar_channels=sar_data.shape[-1], use_mask=True, base_channels=16).to(device)
                        model.eval()
                        if mask_data is None:
                            mask_data = detect_s2_clouds(cloudy_opt_data)

                        opt_t = torch.from_numpy(np.moveaxis(cloudy_opt_data, -1, 0)).unsqueeze(0).float().to(device)
                        sar_t = torch.from_numpy(np.moveaxis(sar_data, -1, 0)).unsqueeze(0).float().to(device)
                        msk_t = torch.from_numpy(np.moveaxis(mask_data, -1, 0)).unsqueeze(0).float().to(device)

                        with torch.no_grad():
                            p_t = model(opt_t, sar_t, msk_t)
                            if use_comp:
                                p_t = composite_cloud_free(p_t, opt_t, msk_t)
                        recon = np.moveaxis(p_t[0].cpu().numpy(), 0, -1)

                        pred_out = type("PredictionOutput", (), {
                            "reconstruction": recon,
                            "cloud_mask": mask_data,
                            "optical_input": cloudy_opt_data,
                            "sar_input": sar_data,
                            "reference": reference_data,
                            "metrics": evaluate_sample_metrics(recon, reference_data, mask_data) if reference_data is not None else None,
                            "metadata": {"compositing_enabled": use_comp, "cloud_fraction": compute_cloud_fraction(mask_data)},
                        })

                    st.markdown("### 📊 Reconstruction Results")

                    # Multi-panel plot
                    fig_res = plot_reconstruction_results(
                        cloudy_optical=pred_out.optical_input,
                        sar=pred_out.sar_input,
                        cloud_mask=pred_out.cloud_mask,
                        reconstruction=pred_out.reconstruction,
                        reference=pred_out.reference,
                    )
                    st.pyplot(fig_res)

                    # Display metrics if reference exists
                    if pred_out.metrics:
                        st.markdown("#### Quality Assessment (vs. Clear Ground Truth)")
                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric("Full PSNR", f"{pred_out.metrics['psnr_full']:.2f} dB" if np.isfinite(pred_out.metrics['psnr_full']) else "Exact")
                        m2.metric("Full SSIM", f"{pred_out.metrics['ssim_full']:.4f}")
                        m3.metric("Full MAE", f"{pred_out.metrics['mae_full']:.4f}")
                        m4.metric("Cloud-Region PSNR", f"{pred_out.metrics.get('psnr_cloud', 0.0):.2f} dB" if pred_out.metrics.get('psnr_cloud') else "N/A")
                        m5.metric("Clear-Region PSNR", f"{pred_out.metrics.get('psnr_clear', 0.0):.2f} dB" if pred_out.metrics.get('psnr_clear') else "N/A")

                    # Downloads
                    d_col1, d_col2 = st.columns(2)
                    with d_col1:
                        out_bytes_io = io.BytesIO()
                        write_satellite_image(pred_out.reconstruction, "outputs/reconstruction.png")
                        with open("outputs/reconstruction.png", "rb") as f_img:
                            st.download_button(
                                "📥 Download Reconstructed Image (PNG)",
                                f_img.read(),
                                "reconstructed_optical.png",
                                "image/png",
                                use_container_width=True,
                            )
                    with d_col2:
                        meta_json = json.dumps(pred_out.metadata, indent=2)
                        st.download_button(
                            "📥 Download Metadata Record (JSON)",
                            meta_json,
                            "reconstruction_metadata.json",
                            "application/json",
                            use_container_width=True,
                        )

                except Exception as exc:
                    st.error(f"Inference failed: {exc}")


# ==============================================================================
# TAB 4: MODEL EVALUATION & BENCHMARKS
# ==============================================================================
elif section == "Model Evaluation & Benchmarks":
    st.markdown('<div class="main-header">Scientific Benchmark Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Independent quantitative comparison across held-out test data.</div>', unsafe_allow_html=True)

    comp_csv = Path("outputs/reports/model_comparison.csv")
    if comp_csv.is_file():
        df_bench = pd.read_csv(comp_csv)
        st.markdown("### 🏆 Comprehensive Model Comparison Table")
        st.dataframe(df_bench, use_container_width=True)

        # Plot bar charts
        summary_records = df_bench.to_dict(orient="records")
        fig_bars = plot_benchmark_bars(summary_records)
        st.pyplot(fig_bars)

        st.download_button(
            "📥 Download Evaluation Results CSV",
            df_bench.to_csv(index=False),
            "model_comparison_results.csv",
            "text/csv",
        )
    else:
        st.info("No saved benchmark comparison found yet. You can run the benchmark script or evaluate existing checkpoints.")
        if st.button("Run Benchmark Evaluation on Test Split"):
            with st.spinner("Evaluating models on held-out test split..."):
                from evaluation.compare_models import run_benchmark_comparison
                ckpts = {
                    "multimodal_unet": "outputs/checkpoints/multimodal_unet.pt",
                    "multimodal_cgan": "outputs/checkpoints/multimodal_cgan.pt",
                    "optical_only": "outputs/checkpoints/optical_only_unet.pt",
                }
                res = run_benchmark_comparison(ckpts, max_eval_samples=25)
                st.success("Benchmark completed! Refresh the page to view charts.")
                st.rerun()


# ==============================================================================
# TAB 5: TRAINING MONITOR
# ==============================================================================
elif section == "Training Monitor":
    st.markdown('<div class="main-header">Model Training & Convergence Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Track training history, loss convergence, and launch configuration-driven training runs.</div>', unsafe_allow_html=True)

    # Check for history files
    reports_dir = Path("outputs/reports")
    hist_files = list(reports_dir.glob("training_history_*.json")) if reports_dir.is_dir() else []

    if hist_files:
        st.markdown("### 📈 Training Loss & Validation History")
        sel_hist = st.selectbox("Select Training History Run:", [f.name for f in hist_files])
        with open(reports_dir / sel_hist, "r") as f:
            hist_data = json.load(f)

        fig_curves = plot_training_curves(hist_data)
        st.pyplot(fig_curves)
        st.json(hist_data)
    else:
        st.info("No completed training history logs found in `outputs/reports/`.")

    st.markdown("---")
    st.markdown("### 🚀 Launch Model Training Run")
    with st.form("train_form"):
        tr_model = st.selectbox("Model Type:", ["multimodal_cgan", "multimodal_unet", "optical_only_unet"])
        tr_epochs = st.number_input("Epochs:", min_value=1, max_value=100, value=2)
        tr_batch = st.number_input("Batch Size:", min_value=1, max_value=32, value=4)
        tr_submit = st.form_submit_button("Start Training Run")

    if tr_submit:
        st.info(f"Training `{tr_model}` for {tr_epochs} epochs with batch size {tr_batch}...")
        with st.spinner("Training in progress..."):
            try:
                if tr_model == "multimodal_cgan":
                    from training.train_gan import train_gan
                    hist = train_gan(epochs_override=tr_epochs, batch_size_override=tr_batch)
                else:
                    from training.train_baseline import train_baseline
                    hist = train_baseline(model_type=tr_model, epochs_override=tr_epochs, batch_size_override=tr_batch)
                st.success(f"Training completed successfully! Model checkpoint saved in `outputs/checkpoints/`.")
                st.rerun()
            except Exception as e:
                st.error(f"Training execution failed: {e}")


# ==============================================================================
# TAB 6: RESEARCH REPORTS & SPECS
# ==============================================================================
elif section == "Research Reports & Specs":
    st.markdown('<div class="main-header">Scientific Research Reports & Provenance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Dataset inspection manifests, sensor specifications, and bibliographic references.</div>', unsafe_allow_html=True)

    insp_file = Path("outputs/reports/dataset_inspection.json")
    if insp_file.is_file():
        st.markdown("### 📋 Dataset Inspection Report")
        with open(insp_file, "r") as f:
            insp_data = json.load(f)
        st.json(insp_data)
        st.download_button("📥 Download Inspection Report (JSON)", json.dumps(insp_data, indent=2), "dataset_inspection.json", "application/json")

    st.markdown("### 📚 Scientific References & Provenance")
    st.markdown(
        """
        1. **SEN12MS-CR-TS Dataset**:
           - Ebel, P., Xu, Y., Schmitt, M., & Zhu, X. X. (2022). *SEN12MS-CR-TS: A Remote-Sensing Data Set for Multimodal Multitemporal Cloud Removal*. IEEE Transactions on Geoscience and Remote Sensing (TGRS). DOI: 10.1109/CVPRW56347.2022.00160.
        2. **Conditional Generative Adversarial Networks (cGANs)**:
           - Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). *Image-to-Image Translation with Conditional Adversarial Networks*. IEEE CVPR.
        3. **Sentinel-1 SAR Principles**:
           - Torres, R., Snoeij, P., Geudtner, D., et al. (2012). *GMES Sentinel-1 mission*. Remote Sensing of Environment, 120, 9-24.
        4. **Sentinel-2 Multispectral Principles**:
           - Drusch, M., Del Bello, U., Carlier, S., et al. (2012). *Sentinel-2: ESA's Optical High-Resolution Mission for GMES Operational Services*. Remote Sensing of Environment, 120, 25-36.
        """
    )
