from js import document, console
from pyodide import create_proxy
from neutron import IncidentNeutron
import time
# import asyncio


from js import document
from io import BytesIO
import pandas as pd

# from pyscript import when

# @when('change', '#upload')
async def _upload_file_and_show(*args):
    csv_file = document.getElementById('upload').files.item(0)

    # array_buf = await csv_file.arrayBuffer() # Get arrayBuffer from file
    # file_bytes = array_buf.to_bytes() # convert to raw bytes array 
    # csv_file = BytesIO(file_bytes) # Wrap in Python BytesIO file-like object

    file = open('csv_file.txt', "w")
    # Write string to file
    file_content = await csv_file.text()
    file.write(file_content)
    # Close the file
    file.close()


    # Read the CSV file into a Pandas DataFrame
    data = IncidentNeutron.from_ace('csv_file.txt')

    # Convert each row to a list and store them in an array
    # rows_array = df.values.tolist()

    # Now, 'rows_array' contains each row of the CSV file as a list
    console.log(data)


# async def _upload_file_and_show(e):
#     console.log("Attempted file upload: " + e.target.value)

#     files = e.target.files.to_py()
#     for file in files:
#       file_content = await file.text()
#       time.sleep(5)

#       f = open("demofile2.txt", "w")
#       f.write(file_content)
#       f.close()

#       data = IncidentNeutron.from_ace("demofile2.txt")

#     newDiv = document.createElement('div')
#     newContent = document.createTextNode(f'success {data}');
#     newDiv.appendChild(newContent);

#     document.getElementById("status_report").appendChild(newDiv)

upload_file = create_proxy(_upload_file_and_show)

document.getElementById("upload").addEventListener("change", upload_file)