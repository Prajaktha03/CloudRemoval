from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from backend.services.auth import SESSION_COOKIE_NAME, AuthStore

router = APIRouter(prefix="/auth", tags=["authentication"])


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)


def get_auth_store(request: Request) -> AuthStore:
    return request.app.state.auth_store


def require_trusted_origin(request: Request) -> None:
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return
    origin = request.headers.get("origin")
    if origin not in request.app.state.settings.frontend_origins:
        raise HTTPException(status_code=403, detail="Request origin is not allowed")


def require_authenticated_user(
    request: Request,
    store: Annotated[AuthStore, Depends(get_auth_store)],
) -> dict[str, object]:
    user = store.get_user(request.cookies.get(SESSION_COOKIE_NAME))
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


@router.post("/login")
def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    store: Annotated[AuthStore, Depends(get_auth_store)],
    _: Annotated[None, Depends(require_trusted_origin)],
):
    source = request.client.host if request.client else "unknown"
    result = store.login(body.email, body.password, source)
    if result is None:
        raise HTTPException(status_code=401, detail="Invalid email or password, or sign-in temporarily limited")
    token, user = result
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=request.app.state.settings.session_max_age_seconds,
        httponly=True,
        secure=request.app.state.settings.session_cookie_secure,
        samesite="strict",
        path="/",
    )
    return {"user": user}


@router.post("/logout")
def logout(
    request: Request,
    response: Response,
    store: Annotated[AuthStore, Depends(get_auth_store)],
    _: Annotated[None, Depends(require_trusted_origin)],
    __: Annotated[dict[str, object], Depends(require_authenticated_user)],
):
    store.logout(request.cookies.get(SESSION_COOKIE_NAME))
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        secure=request.app.state.settings.session_cookie_secure,
        samesite="strict",
    )
    return {"ok": True}


@router.get("/me")
def current_user(
    user: Annotated[dict[str, object], Depends(require_authenticated_user)],
):
    return {"user": user}
