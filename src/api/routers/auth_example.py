"""Example authentication router using simplified security service.

This is an example implementation showing how to use the security service.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from src.services.security import (
    TokenResponse,
    UserCredentials,
    create_access_token,
    hash_password,
    require_auth,
    verify_password,
)


router = APIRouter(prefix="/auth", tags=["Authentication"])

# Example user store (in production, use a proper database)
USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": hash_password("admin123"),  # Example only!
        "is_active": True,
    }
}


@router.post("/login", response_model=TokenResponse)
async def login(credentials: UserCredentials):
    """Authenticate user and return JWT token."""
    # Get user from database
    user = USERS_DB.get(credentials.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    # Verify password
    if not verify_password(
        credentials.password.get_secret_value(), user["hashed_password"]
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    # Check if user is active
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )

    # Create access token
    token = create_access_token(subject=credentials.username)

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=24 * 3600,  # 24 hours in seconds
    )


@router.get("/me")
async def get_current_user(user_id: str = Depends(require_auth)):
    """Get current authenticated user information."""
    user = USERS_DB.get(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Don't return the hashed password
    return {
        "username": user["username"],
        "is_active": user.get("is_active", True),
    }


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(credentials: UserCredentials, _: str = Depends(require_auth)):
    """Register a new user (admin only)."""
    # Check if user already exists
    if credentials.username in USERS_DB:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User already exists",
        )

    # Create new user
    USERS_DB[credentials.username] = {
        "username": credentials.username,
        "hashed_password": hash_password(credentials.password.get_secret_value()),
        "is_active": True,
    }

    return {"message": f"User {credentials.username} created successfully"}


# To use this router in main.py:
# from .routers import auth_example
# app.include_router(auth_example.router, prefix="/api/v1")
