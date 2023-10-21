from js import document, console
from pyodide import create_proxy
from neutron import IncidentNeutron
# import asyncio

async def _upload_file_and_show(e):
    console.log("Attempted file upload: " + e.target.value)

    
    files = e.target.files.to_py()
    for file in files:
      file_content = await file.text()
      data = IncidentNeutron.from_ace(file_content)


    newDiv = document.createElement('div')
    newContent = document.createTextNode(f'success {data}');
    newDiv.appendChild(newContent);

    document.getElementById("status_report").appendChild(newDiv)

upload_file = create_proxy(_upload_file_and_show)

document.getElementById("file-upload").addEventListener("change", upload_file)