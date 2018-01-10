window.onload = main;

function main() {
    const ws = new WebSocket("ws://127.0.0.1:5678/");
    let prev;

    ws.onmessage = event => {
        const data = JSON.parse(event.data);

        if (!prev) {
            initWorld(data.world);
        } else {
            fillCell(prev.drone, '');
            fillCell(prev.target, '');
        }

        fillCell(data.drone, 'drone');
        fillCell(data.target, 'target');

        prev = data;
    };
}

function initWorld(worldSize) {
    const table = document.getElementById('table');
    for (let r = 0; r < worldSize[0]; r++) {
        const row = document.createElement('tr');
        for (let c = 0; c < worldSize[0]; c++) {
            const cell = document.createElement('td');
            cell.id = `${r}:${c}`;
            row.appendChild(cell);
        }
        table.appendChild(row);
    }
}

function fillCell(cell, className) {
    document.getElementById(`${cell[0]}:${cell[1]}`).className = className;
}