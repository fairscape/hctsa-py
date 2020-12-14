from minio import Minio
import sys

file_name = sys.argv[1]

folder = 'Non-PreVent-Vitals/'

# minioClient = Minio('127.0.0.1:9000',
#                     access_key='92WUKA7ZAP4M3UOS0TNG',
#                     secret_key='uIgJzgatEyop9ZKWfRDSlgkAhDtOzJdF+Jw+N9FE',
#                     secure=False)
minioClient = Minio('minionas.int.uvadcos.io',
                    access_key='breakfast',
                    secret_key='breakfast',
                    secure=False)
#print(folder + file_name)
data = minioClient.get_object("prevent", folder + file_name)
with open(file_name, 'wb') as file_data:
        for d in data.stream(32*1024):
            file_data.write(d)
