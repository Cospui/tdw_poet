from json import loads
from os.path import isdir, join
from os import mkdir, getcwd

directory = join(getcwd(), "dist")
object_data_recv = False
recieved_data = None
save_data = False
save_dir = None
image_index = 0

def on_recv(messages):
    """
    Example on_recv callback.

    :param messages: The received messages.
    """

    # Create the image save directory.
    global object_data_recv
    global recieved_data
    global save_data
    global image_index
    global directory
    tdw.global_message_count += 1

    for message_dict in messages:
        # Convert the byte array to a JSON object.
        message_dict = loads(message_dict)
        if "objects_data" in message_dict["$type"]:
            recieved_data = message_dict
            object_data_recv = True
        # Grab image data and print meta-data.
        if message_dict["$type"] == "image_data":
            # Print the metadata.

            # for key in message_dict:
            #     if key == "$type" or key == "passes":
            #         continue
            #     elif key == "time":
            #         print(key + ": " + tdw.get_time(message_dict[key]))
            #     else:
            #         print(key + ": " + str(message_dict[key]))

            # Print the passes and output the images.
            for image_pass in message_dict["passes"]:
                # for pass_key in image_pass:
                    # if pass_key == "$type" or pass_key == "extension" or pass_key == "image":
                    #     continue
                    # else:
                    #     print(pass_key + ": " + str(image_pass[pass_key]))
                # Save the image.
                if message_dict["sensor_name"] in "SensorContainer":
                    path = join(save_dir, "image-{0}".format(image_index) + image_pass["extension"])
                else:
                    path = join(save_dir, "image-{0}".format(image_index) + image_pass["extension"])
                image_index += 1
                tdw.save_img(image_pass["image"], path)

                # print("Saved image: " + path)
        # else:
            # Iterate through each entry in the message.
            # for key in message_dict.keys():
            #     if key == "$type":
            #         continue
            #     elif key == "time":
            #         print(key + ": " + tdw.get_time(message_dict[key]))
            #     else:
            #         print(key + ": " + str(message_dict[key]))
    # Send an empty message.
    # tdw.send_to_server({"$type": "step_physics", "frames": 0})
