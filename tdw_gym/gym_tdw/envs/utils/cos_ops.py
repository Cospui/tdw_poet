import boto3
import pandas as pd
import os

class port_tracker:
    def __init__(self):
        cos = boto3.resource(
            service_name="s3",
            aws_access_key_id="e40b956720ad494f82db225b0980b984",
            aws_secret_access_key="468361dddc271d9222c50f5f25824057262d7c08927340a7",
            endpoint_url="https://s3-api.us-geo.objectstorage.softlayer.net"
        )
        for bucket in cos.buckets.all():
           if bucket.name in 'tdw':
               self.tdw_bucket = bucket
        self.port_tracker = None
        self.ip = self.in_port = self.out_port = self.ID = None

    def download_file(self, filename):
        self.tdw_bucket.download_file(filename, filename)
        self.port_tracker = pd.read_csv(filename)

    def upload_file(self, filename):
        self.tdw_bucket.upload_file(filename, filename)
        os.remove(filename)

    def delete_file(self, filename):
        self.tdw_bucket.download_file(  Delete={
        'Objects': [
            {
                'Key': 'tracker.csv'
            },
        ],
        'Quiet': True|False
    },
    MFA='string',
    RequestPayer='requester',
    BypassGovernanceRetention=True|False)
    def get_ports(self):
        try:
            self.download_file("tracker.csv")
            self.port_tracker = pd.read_csv('tracker.csv')
            os.remove("tracker.csv")
        except:
            print("Tracker file does not exist. Creating one now.")
            self.port_tracker = pd.DataFrame(columns=["ID", "port", "tracker_status"])
            port = 1071
            for i in range(4):
                self.port_tracker.loc[i] = [i, "52.116.149.123", port, port + 1, "free"]
                port += 2
            self.port_tracker.to_csv('tracker.csv', index=False)
            self.upload_file("tracker.csv")
        for i in range(self.port_tracker.shape[0]):
            if self.port_tracker["tracker_status"].iloc[i] == "free":
                self.ip = self.port_tracker["ip"].iloc[i]
                self.in_port = int(self.port_tracker["in_port"].iloc[i])
                self.out_port = int(self.port_tracker["out_port"].iloc[i])
                self.ID = self.port_tracker["ID"].iloc[i]
                self.port_tracker["tracker_status"].iloc[i] = "not_free"
                self.port_tracker.to_csv('tracker.csv', index=False)
                self.upload_file('tracker.csv')
                return
        raise Exception("There are not free ports available at this moment. Makesure to use env.close() at end of your script to free up the ports")

    def free_up_port(self):
        self.download_file("tracker.csv")
        self.port_tracker = pd.read_csv('tracker.csv')
        os.remove("tracker.csv")
        for i in range(self.port_tracker.shape[0]):
            if self.ID == self.port_tracker["ID"].iloc[i]:
                self.port_tracker["tracker_status"].iloc[i] = "free"
                self.port_tracker.to_csv('tracker.csv', index=False)
                self.upload_file('tracker.csv')
                print("Port has been freed up")
                return
        raise Exception("The port you are using is not in the tracker file")
