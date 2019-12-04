#!/usr/bin/env python
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
from pymongo import MongoClient
from bson.json_util import dumps, loads
from bson import ObjectId
import argparse
from urllib.parse import parse_qs, unquote


class S(BaseHTTPRequestHandler):
    """
    The server class.
    """

    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        # Fetch model info by ID.
        if str(self.path).startswith("/fetch_model_info_by_id"):
            msg = fetch_model_info_by_id(self.path)
        # Fetch model info by name.
        elif str(self.path).startswith("/fetch_model_info_by_name"):
            msg = fetch_model_info_by_name(self.path)
        # Fetch all wnids.
        elif str(self.path).startswith("/fetch_all_wnid"):
            msg = fetch_all_wnids()
        # Fetch all models with a given wnid.
        elif str(self.path).startswith("/fetch_models_by_wnid"):
            msg = fetch_models_by_wnid(self.path)
        elif str(self.path).startswith("/fetch_all_wcategories"):
            msg = fetch_all_categories()
        elif str(self.path).startswith("/fetch_models_info_by_names"):
            msg = fetch_models_by_names(self.path)
        elif str(self.path).startswith("/fetch_material_by_name"):
            msg = fetch_material_by_name(self.path)
        elif str(self.path).startswith("/fetch_all_materials"):
            msg = fetch_all_materials()
        elif str(self.path).startswith("/fetch_hdri_by_name"):
            msg = fetch_hdri_by_name(self.path)
        elif str(self.path).startswith("/fetch_all_hdris"):
            msg = fetch_all_hdris()
        elif str(self.path).startswith("/fetch_all_model_ids"):
            msg = fetch_all_model_ids()
            msg = loads(dumps(msg))
            msg = [str(m["_id"]) for m in msg]
        else:
            msg = "ERROR!!!"

        b = bytearray()
        b.extend(map(ord, dumps(msg)))
        self.wfile.write(b)

    def do_GET(self):
        if not quiet_mode:
            logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self._set_response()

    def do_POST(self):
        length = int(self.headers['content-length'])
        form = parse_qs(self.rfile.read(length), keep_blank_values=1)
        if form[b'action'][0] == b'set_substructure':
            set_substructure(form)


def set_substructure(form):
    """
    Set the substructure of the object.

    :param form: The data form.
    """

    record_id = ObjectId(form[b'id'][0].decode('utf-8'))
    substructure = unquote(form[b'value'][0].decode('utf-8'))
    MODELS_COLLECTION.find_one_and_update({"_id": record_id}, {"$set": {"substructure": substructure}})


def get_models_collection():
    """
    Returns the models collection from the database.
    """

    client = MongoClient(connection_string)
    db = client['model_lib']
    collection = db['models']
    return collection


def get_materials_collection():
    """
    Returns the materials collection from the database.
    """

    client = MongoClient(connection_string)
    db = client['model_lib']
    collection = db['materialTypes']
    return collection


def get_hdris_collection():
    """
    Returns the hdri collection from the database.
    """

    client = MongoClient(connection_string)
    db = client['model_lib']
    collection = db['skyboxes']
    return collection


def fetch_model_info_by_id(path):
    """
    Fetch a model, given its database id.

    :param path: The URL path, including the ID.
    """

    return MODELS_COLLECTION.find_one(filter={"_id": ObjectId(str(path).split("?=")[1])})


def fetch_model_info_by_name(path):
    """
    Fetch a model, given its name.

    :param path: The URL path, including the name.
    """

    # Parse spacing properly.
    name = str(path).split("?=")[1].replace("%C2%A0", " ")
    return MODELS_COLLECTION.find_one(filter={"model_name": name})


def fetch_models_by_wnid(path):
    """
    Fetch all models with a given WordNet ID.

    :param path: The URL path, including the WordNet ID.
    """

    return MODELS_COLLECTION.find(filter={"wnid": str(path).split("?=")[1]})


def fetch_all_wnids():
    """
    Fetch all WordNet IDs in the database.
    """

    return MODELS_COLLECTION.find(projection={"wnid" : True, "_id" : False})


def fetch_all_categories():
    """
    Fetch all WordNet categories in the database.
    """

    return MODELS_COLLECTION.find(projection={"wcategory": True, "wnid": True, "_id": False})


def fetch_models_by_names(path):
    """
    Given the names of models, fetch the records.

    :param path: The URL path, with model names separated by semicolons: /fetch_models_info_by_names?=some_chair;a_desk
    """

    names_args = str(path).split("?=")[1].split(";")
    records = []
    for n in names_args:
        n = n.replace("%20", " ")
        records.append(MODELS_COLLECTION.find_one(filter={"model_name": n}))
    return records


def fetch_all_model_ids():
    """
    Returns a list of all model IDs.
    """

    return MODELS_COLLECTION.find(projection={"_id.ObjectID"})


def fetch_material_by_name(path):
    """
    Fetch a material, given its name.

    :param path: The URL path, including the name.
    """

    name = str(path).split("?=")[1].replace("%C2%A0", " ")
    return MATERIALS_COLLECTION.find_one(filter={"material_name": name})


def fetch_all_materials():
    """
    Fetches all materials.
    """

    return MATERIALS_COLLECTION.find()

	
def fetch_hdri_by_name(path):
    """
    Fetch an hdri, given its name.

    :param path: The URL path, including the name.
    """

    name = str(path).split("?=")[1].replace("%C2%A0", " ")
    return HDRIS_COLLECTION.find_one(filter={"skybox_name": name})


def fetch_all_hdris():
    """
    Fetches all materials.
    """

    return HDRIS_COLLECTION.find()


def run(port):
    """
    Run db_server forever.

    :param port: The port to run db_server on.
    """

    if not quiet_mode:
        logging.basicConfig(level=logging.INFO)
    server_address = ('127.0.0.1', port)
    httpd = HTTPServer(server_address, S)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    if not quiet_mode:
        logging.info('Stopping httpd...\n')


connection_string = "mongodb://tdwuser:tdw2018@ds159200.mlab.com:59200/model_lib"
MODELS_COLLECTION = get_models_collection()
MATERIALS_COLLECTION = get_materials_collection()
HDRIS_COLLECTION = get_hdris_collection()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # The port for connecting with the build.
    parser.add_argument('-p', dest='server_port', default=8080, type=int, metavar='Port', help="Default=8080")
    #  Connection string used to connect to the server.
    parser.add_argument('-o', dest='connection_string',
                        default="mongodb://tdwuser:tdw2018@ds159200.mlab.com:59200/model_lib",
                        type=str, metavar='Connection string', help="")
    # Toggle for quiet mode.
    parser.add_argument('-q', dest='quiet_mode', action='store_true', help='Quiet')
    # Parse the arguments.
    args = parser.parse_args()

    server_port = args.server_port
    connection_string = args.connection_string
    quiet_mode = args.quiet_mode

    run(server_port)
