"""Account administration, run inside the backend container:

    docker compose exec backend python -m app.cli create-user someone@example.org --name "Dr Someone"
    docker compose exec backend python -m app.cli reset-password someone@example.org
    docker compose exec backend python -m app.cli list-users
"""

import argparse
import getpass
import sys

from sqlalchemy import delete, func, select

from app.db import SessionLocal, init_db
from app.models import AuthSession, Document, Query, User
from app.routers.auth import _claim_legacy_data
from app.schemas import PASSWORD_MIN
from app.security import hash_password


def _password() -> str:
    while True:
        password = getpass.getpass("Password: ")
        if len(password) < PASSWORD_MIN:
            print(f"Use at least {PASSWORD_MIN} characters.")
        elif getpass.getpass("Repeat password: ") != password:
            print("The passwords don't match.")
        else:
            return password


def main() -> None:
    parser = argparse.ArgumentParser(description="MedRAG account administration")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create-user", help="add an account (works when registration is disabled)")
    create.add_argument("email")
    create.add_argument("--name")
    reset = sub.add_parser("reset-password", help="set a new password and sign the user out everywhere")
    reset.add_argument("email")
    sub.add_parser("list-users")
    args = parser.parse_args()

    init_db()
    with SessionLocal() as db:
        if args.command == "list-users":
            for user in db.scalars(select(User).order_by(User.id)):
                docs = db.scalar(select(func.count()).select_from(Document).where(Document.user_id == user.id))
                questions = db.scalar(select(func.count()).select_from(Query).where(Query.user_id == user.id))
                print(f"{user.id:4}  {user.email:40} {user.name or '':20} {docs} documents, {questions} questions")
            return

        email = args.email.strip().lower()
        user = db.scalar(select(User).where(User.email == email))
        if args.command == "create-user":
            if user is not None:
                sys.exit(f"{email} already exists")
            first = db.scalar(select(func.count()).select_from(User)) == 0
            user = User(email=email, name=args.name, password_hash=hash_password(_password()))
            db.add(user)
            db.flush()
            if first:
                _claim_legacy_data(db, user)
            db.commit()
            print(f"Created {email}")
        else:
            if user is None:
                sys.exit(f"No account for {email}")
            user.password_hash = hash_password(_password())
            db.execute(delete(AuthSession).where(AuthSession.user_id == user.id))
            db.commit()
            print(f"Password changed for {email}; existing sessions were signed out")


if __name__ == "__main__":
    main()
