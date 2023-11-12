from sqlalchemy import create_engine, ForeignKey, Column, String, Integer, CHAR
from sqlalchemy.orm import sessionmaker, declarative_base
from pydantic import BaseModel

BaseSql = declarative_base()


class Person(BaseModel):
    ssn: int
    firstname: str
    lastname: str
    gender: str
    age: int


class PersonModel(BaseSql):
    __tablename__ = "people"

    ssn = Column("ssn", Integer, primary_key=True)
    firstname = Column("firstname", String)
    lastname = Column("lastname", String)
    gender = Column("gender", CHAR)
    age = Column("age", Integer)

    def __init__(self, ssn, first, last, gender, age):
        self.ssn = ssn
        self.firstname = first
        self.lastname = last
        self.gender = gender
        self.age = age

    def __repr__(self):
        return Person(
            ssn=self.ssn,
            firstname=self.firstname,
            lastname=self.lastname,
            gender=self.gender,
            age=self.age,
        )


class ThingModel(BaseSql):
    __tablename__ = "things"

    tid = Column("tid", Integer, primary_key=True)
    description = Column("description", String)
    owner = Column(Integer, ForeignKey("people.ssn"))

    def __init__(self, tid, description, owner):
        self.tid = tid
        self.description = description
        self.owner = owner

    def __repr__(self):
        return f"({self.tid}) {self.description} owned by {self.owner}"


engine = create_engine("sqlite:///mydb.db", echo=False)

# INFO Creates all the tables using the `engine` and BaseSql
BaseSql.metadata.create_all(bind=engine)
Session = sessionmaker(bind=engine)
session = Session()

person1 = PersonModel(123123, "Mike", "Smith", "m", 35)
session.add(person1)
session.commit()

person2 = PersonModel(123321, "A", "Smith", "m", 25)
person3 = PersonModel(123, "B", "Smith", "f", 19)
person4 = PersonModel(321, "C", "Smith", "f", 8)
session.add(person2)
session.add(person3)
session.add(person4)
session.commit()

# results = session.query(PersonModel).filter(PersonModel.firstname == "A")
results = session.query(PersonModel).filter(PersonModel.age <= 20)
for r in results:
    print(type(r.age))
exit()
# results = session.query(Person).filter(Person.firstname.like("%An%"))
# results = session.query(Person).filter(Person.firstname.in_(["A", "B"]))

# for r in results:
#     print(r)

t1 = Thing(1, "Car", person1.ssn)
t2 = Thing(2, "Ps4", person1.ssn)
t3 = Thing(3, "Boombox", person2.ssn)
t4 = Thing(4, "eodfn", person3.ssn)
t5 = Thing(5, "laptop", person3.ssn)
session.add(t1)
session.add(t2)
session.add(t3)
session.add(t4)
session.add(t5)
session.commit()

results = (
    session.query(Thing, Person)
    .filter(Thing.owner == Person.ssn)
    .filter(Person.firstname.like("A%"))
)
for r in results:
    print(r)
