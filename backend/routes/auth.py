from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.auth.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from backend.database.database import get_db
from backend.database.models import User


router = APIRouter()


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post(
    "/register",
    summary="Register a new user",
    description="Creates a new user account with a securely hashed password.",
)
def register_user(
    request: RegisterRequest,
    db: Session = Depends(get_db),
):
    username = request.username.strip()

    if not username:
        raise HTTPException(
            status_code=400,
            detail="Username cannot be empty.",
        )

    if not request.password:
        raise HTTPException(
            status_code=400,
            detail="Password cannot be empty.",
        )

    existing_user = (
        db.query(User)
        .filter(User.username == username)
        .first()
    )

    if existing_user:
        raise HTTPException(
            status_code=409,
            detail="Username already exists.",
        )

    hashed_password = hash_password(request.password)

    user = User(
        username=username,
        password_hash=hashed_password,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "message": "User registered successfully.",
        "username": user.username,
    }


@router.post(
    "/login",
    summary="Log in a user",
    description=(
        "Authenticates a user and returns a JWT access token "
        "for protected API endpoints."
    ),
)
def login_user(
    request: LoginRequest,
    db: Session = Depends(get_db),
):
    username = request.username.strip()

    user = (
        db.query(User)
        .filter(User.username == username)
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
        )

    password_valid = verify_password(
        request.password,
        user.password_hash,
    )

    if not password_valid:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
        )

    access_token = create_access_token(
        data={"sub": user.username}
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
    }