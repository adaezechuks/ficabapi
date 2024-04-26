from flask import request, jsonify,abort
from config import app, db
from models import Reviews,ReviewDetail
from Predicting import Predicting

@app.route('/get-review-by-id/<int:id>', methods=['GET'])
def get_review_by_id(id):
    review = Reviews.query.get(id)
    if not review:
        return jsonify({"error": "Review not found"}), 404
    
    review_data = review.to_json()

    details = ReviewDetail.query.filter_by(review_id=id).all()
    review_data['details'] = [detail.to_json() for detail in details]

    return jsonify(review_data), 200

@app.route('/get-all-reviews', methods=['GET'])
def get_all_reviews():
    # Fetch all reviews from the database
    # reviews = Reviews.query.all()
    reviews = Reviews.query.order_by(Reviews.id.desc()).all()

    reviews_list = []

    # Convert each review to JSON format and include details
    for review in reviews:
        review_data = review.to_json()
        
        # Fetch related review details and convert to JSON
        details = ReviewDetail.query.filter_by(review_id=review.id).all()
        review_data['details'] = [detail.to_json() for detail in details]
        
        reviews_list.append(review_data)

    return jsonify(reviews_list), 200

@app.route('/predict', methods=['POST'])
def predict():
    reqbody = request.json
    review_text = reqbody.get('review')
    rating = reqbody.get('rating', None)  # Optionally get rating if available
    customer=reqbody.get('customer',None) #optional customer name

    if not review_text:
        return jsonify({"error": "Please provide a review"}), 400

    # Load or train model and predict opinion
    predictor = Predicting()
    predictor.load_or_train_model('trialdata.xml')
    opinion = predictor.predict_opinion(review_text)

    # Insert the review into the Reviews table
    new_review = Reviews(review_text=review_text, rating=rating,customer=customer)
    db.session.add(new_review)
    db.session.commit()


    # Insert details into ReviewDetail table
    for detail in opinion:
        new_detail = ReviewDetail(review_id=new_review.id, target=detail[0], polarity=detail[1])
        db.session.add(new_detail)
        print(new_detail)

    db.session.commit()  # Commit all details at once

    return jsonify({"result": opinion}), 200


@app.route('/reviews/<int:review_id>', methods=['DELETE'])
def delete_review(review_id):
    review = Reviews.query.get_or_404(review_id)  # Get the review or return 404 if not found
    db.session.delete(review)  # Delete the review
    db.session.commit()  # Commit the transaction
    return jsonify({'result': 'Review and its details successfully deleted'}), 200








# Create a new review
@app.route('/reviews', methods=['POST'])
def create_review():
    if not request.json or not 'polarity' in request.json or not 'reviewText' in request.json:
        abort(400)  # Missing information
    review = Reviews(
        polarity=request.json['polarity'],
        review_text=request.json['reviewText'],
        customer = request.json['customer']
    )
    db.session.add(review)
    db.session.commit()
    return jsonify(review.to_json()), 201

# Get all reviews
@app.route('/reviews', methods=['GET'])
def get_reviews():
    reviews = Reviews.query.all()
    return jsonify([review.to_json() for review in reviews])

# Get a specific review
@app.route('/reviews/<int:review_id>', methods=['GET'])
def get_review(review_id):
    review = Reviews.query.get_or_404(review_id)
    return jsonify(review.to_json())

# Update a review
@app.route('/reviews/<int:review_id>', methods=['PUT'])
def update_review(review_id):
    review = Reviews.query.get_or_404(review_id)
    if not request.json:
        abort(400)
    
    review.polarity = request.json.get('polarity', review.polarity)
    review.review_text = request.json.get('reviewText', review.review_text)
    db.session.commit()
    return jsonify(review.to_json())




if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(debug=True)