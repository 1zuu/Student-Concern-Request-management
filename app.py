import json
from pymongo import MongoClient
from bson.objectid import ObjectId
from flask import Flask, Response, request
from inference import SCRM_Inference
from variables import*

app = Flask(__name__)

try:
    client = MongoClient(db_url)
    db = client[database]
    client.server_info()

except:
    print("Error: Unable to connect to database.")

inf = SCRM_Inference()
inf.data_to_features()
inf.nearest_neighbor_model()

@app.route("/insert", methods=["POST"])
def insert():
    try:
        concerns_data = request.get_json()
        cursor = db[live_collection].find()
        data = list(cursor)
        if len(data) == 0:
           concerns_data['student_id'] = str(1)
        else:
           concerns_data['student_id'] = str(int(data[-1]['student_id']) + 1)
        
        dbResponse = db[live_collection].insert_one(concerns_data)
        return Response(
                    response=json.dumps({
                        "status": "success",
                        "id" : str(dbResponse.inserted_id)
                            }), 
                    status=200, 
                    mimetype="application/json"
                    )

    except Exception as e:
        print(e)

@app.route("/get", methods=["GET"])
def get():
    try:
        cursor = db[live_collection].find()
        data = list(cursor)
        for param in data:
            param["_id"] = str(param["_id"])
        return Response(
                    response=json.dumps(data), 
                    status=500, 
                    mimetype="application/json"
                    )


    except Exception as e:
        print(e)
        return Response(
                    response=json.dumps({
                        "status": "unsuccessful"
                            }), 
                    status=500, 
                    mimetype="application/json"
                    )

@app.route("/update", methods=["PATCH"])
def update():
    try:
        data = list(db[live_collection].find())[-1]
        obj_id = str(data["_id"])

        concern_data = request.get_json()
        response = inf.make_response(concern_data)
        dbResponse = db[live_collection].update_one(
                                        {"_id": ObjectId(obj_id)}, 
                                        {"$set": response}
                                             )
        return Response(
                    response=json.dumps({
                        "status": response
                            }), 
                    status=200, 
                    mimetype="application/json"
                    )
    except Exception as e:
        print(e)
        return Response(
                    response=json.dumps({
                        "status": "can not update user {}".format(id)
                            }), 
                    status=500, 
                    mimetype="application/json"
                    )

if __name__ == '__main__':
    app.run(debug=True, host=host)