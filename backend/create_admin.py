from getpass import getpass

from backend.core.config import settings
from backend.services.auth import AuthStore, MAX_PASSWORD_LENGTH, MIN_PASSWORD_LENGTH


def main() -> None:
    email = input("Account email: ").strip()
    password = getpass(f"Password ({MIN_PASSWORD_LENGTH}-{MAX_PASSWORD_LENGTH} characters): ")
    confirmation = getpass("Confirm password: ")
    if password != confirmation:
        raise SystemExit("Passwords do not match")

    store = AuthStore(settings.database_path, settings.session_max_age_seconds)
    try:
        store.create_user(email, password)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Account created for {email.strip().lower()}")


if __name__ == "__main__":
    main()
