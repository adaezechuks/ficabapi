from config import db
from datetime import datetime


class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(80), unique=False, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    last_name = db.Column(db.String(80), unique=False, nullable=False)

    def to_json(self):
        return {
            "id" : self.id,
            "firstName": self.first_name,
            "lastName":self.last_name,
            "email":self.email,
        }
    
class Reviews(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    rating = db.Column(db.Integer, unique=False, nullable=True)
    review_time = db.Column(db.DateTime, unique=False, nullable=False, default=datetime.now)
    customer = db.Column(db.String(100), unique=False, nullable=True)
    review_text = db.Column(db.String(500), unique=False, nullable=False)

    # Relationship to link Reviews with ReviewDetail
    details = db.relationship('ReviewDetail', backref='review', lazy=True, cascade="all, delete-orphan")


    def to_json(self):
        return {
            "id": self.id,
            "rating": self.rating,
            "reviewTime": self.review_time.strftime('%Y-%m-%d %H:%M:%S'),
            "reviewText": self.review_text,
            "customer": self.customer,
        }

class ReviewDetail(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    review_id = db.Column(db.Integer, db.ForeignKey('reviews.id'), nullable=False)
    target = db.Column(db.String(100), nullable=False)
    polarity = db.Column(db.String(20), nullable=False)

    def to_json(self):
        return {
            "id": self.id,
            "reviewId": self.review_id,
            "target": self.target,
            "polarity": self.polarity
        }