import asyncio
from datetime import datetime
from pymongo import MongoClient
import json
import hashlib
from bson.json_util import dumps


def parse_json(data):
    return json.loads(dumps(data))


def getUsersFromDB():
    client = MongoClient(
        'mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false')
    db = client.MLChat
    users = parse_json(list(db.Users.find()))
    # print(users)
    return users


def insertUserToDB(user):
    client = MongoClient(
        'mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false')
    db = client.MLChat
    users = db.Users
    userJson = {
        "username": user['username'],
        "password": hashlib.md5(user['password'].encode()).hexdigest(),
        "friends": []}
    users.insert_one(userJson)
    return


def checkUserExistsInDB(user):
    client = MongoClient(
        'mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false')
    db = client.MLChat
    users = db.Users
    storedUser = users.find_one({'username': user['username']})
    if storedUser:
        return True
    else:
        return False


def addFriendToUserInDB(user):
    client = MongoClient(
        'mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false')
    db = client.MLChat
    users = db.Users
    updateUser = users.find_one_and_update({'username': user['username']}, {
                                           "$push": {"friends": user['friend']}})
    if(updateUser):
        return True
    else:
        return False


def removeFriendFromUserInDB(user):
    client = MongoClient(
        'mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false')
    db = client.MLChat
    users = db.Users
    updateUser = users.find_one_and_update({'username': user['username']}, {
                                           "$pull": {"friends": user['friend']}})
    if(updateUser):
        return True
    else:
        return False


def getUsersByUsernameInDB(user):
    client = MongoClient(
        'mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false')
    db = client.MLChat
    users = db.Users
    storedUser = users.find(
        {'username': {"$regex": user['username'], "$options": 'i'}})
    if storedUser:
        return parse_json(list(storedUser))
    else:
        return False


def sendMessageInDB(user):  # TODO: need to check if spam
    client = MongoClient(
        'mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false')
    db = client.MLChat
    messages = db.pendingMessages
    sentMessage = messages.insert_one({"from": user['username'],
                                       "to": user['reciever'],
                                       "timestamp": datetime.now(),
                                       "content": user['content']})
    if sentMessage:
        return True
    else:
        return False


async def recieveMessageFromDB(user):
    client = MongoClient(
        'mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false')
    result = []
    db = client.MLChat
    messages = db.pendingMessages
    result = messages.find({"to": user['username']})
    if result:
        return parse_json(list(result))
    else:
        return False

async def deleteRecievedMessages(user):
    client = MongoClient(
        'mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false')
    result = []
    db = client.MLChat
    messages = db.pendingMessages
    result = messages.delete_many({"to": user['username']})
    if result:
        return True
    else:
        return False