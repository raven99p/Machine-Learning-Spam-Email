from dao import *
from flask import Flask, request
from flask import jsonify


app = Flask(__name__)


@app.route('/getUsers')
def getUsers():
    myUsers = getUsersFromDB()
    return jsonify(myUsers)


@app.route('/register', methods=["POST"])
def register():
    if request.method == "POST":
        insertUserToDB(request.form)
    return jsonify({"message": "ok"})
    # registerUserInDB()


@app.route('/checkUser', methods=["POST"])
def checkUser():
    if request.method == "POST":
        userExists = checkUserExistsInDB(request.form)
        return jsonify({"userExists": userExists})


@app.route('/addFriend', methods=["POST"])
def addFriend():
    if request.method == "POST":
        addedFriend = addFriendToUserInDB(request.form)
        return jsonify({"added": addedFriend})


@app.route('/removeFriend', methods=["POST"])
def removeFriend():
    if request.method == "POST":
        addedFriend = removeFriendFromUserInDB(request.form)
        return jsonify({"removed": addedFriend})
    
@app.route('/searchUsersByUsername', methods=["POST"])
def getUserByUsername():
    if request.method == "POST":
        requestedUser = getUsersByUsernameInDB(request.form)
        return jsonify({"requestedUser": requestedUser})
    
    
@app.route('/sendMessage', methods=["POST"])
def sendMessage():
    if request.method == "POST":
        print(request.form)
        messageSent = sendMessageInDB(request.form)
        return jsonify({"messageSent": messageSent})
    
    
@app.route('/recieveMessages', methods=["POST"])
async def recieveMessages():
    if request.method == "POST":
        messages = await recieveMessageFromDB(request.form)
        deleted = await deleteRecievedMessages(request.form)
        return jsonify({"messages": messages})