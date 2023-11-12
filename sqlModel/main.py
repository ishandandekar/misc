from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine


class Movies(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    genre: str
    sensitive: Optional[bool] = False


movie_1 = Movies(name="La a land", genre="musical", sensitive=False)

engine = create_engine("sqlite:///mydb.db")

SQLModel.metadata.create_all(engine)

with Session(engine) as session:
    session.add(movie_1)
    session.commit()
