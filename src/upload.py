from js import document, console
from pyodide import create_proxy
from ace import IncidentNeutron


async def _upload_file_and_show(*args):
    csv_file = document.getElementById('upload').files.item(0)
    with open('csv_file.txt', "w") as file:
        file_content = await csv_file.text()
        file.write(file_content)

    try:
        data = IncidentNeutron.from_ace('csv_file.txt')

        console.log(data.reactions)

        # newDiv = document.createElement('div')
        # newContent1 = document.createTextNode(f'success');
        # newContent2 = document.createTextNode(f'{data.reactions}');
        # newDiv.appendChild(newContent1);
        # newDiv.appendChild(newContent2);
        document.getElementById("status_report").textContent = f'success {data.reactions}'
    except:

        document.getElementById("status_report").textContent = 'conversion failed'


upload_file = create_proxy(_upload_file_and_show)

document.getElementById("upload").addEventListener("change", upload_file)