from sqlalchemy import Column, Integer, String

from backend.database.database import Base


class User(Base):
    """
    Database model representing an application user.
    """

    __tablename__ = "users"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    username = Column(
        String,
        unique=True,
        index=True,
        nullable=False
    )

    password_hash = Column(
        String,
        nullable=False
    )