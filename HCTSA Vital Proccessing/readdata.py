#© 2020 By The Rector And Visitors Of The University Of Virginia

#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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
