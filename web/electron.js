document.onkeydown = updateKey;
document.onkeyup = resetKey;

var server_port = 8080;
var server_addr = "172.16.103.6";   // the IP address of your Raspberry PI

function client(){
    
    const net = require('net');
    fetch("http://"+server_addr+":"+server_port+"/pose_slam")
        .then(rs => rs.json())
        .then(dat => {
            document.getElementById("heading").textContent = dat.heading;
            document.getElementById("speed").textContent = dat.v;
        });

    fetch("http://"+server_addr+":"+server_port+"/planner_status")
        .then(rs => rs.json())
        .then(dat => {
            document.getElementById("mode").textContent = dat.mode;
        });
}

function sendVelocity(v_command, w_command) {
    let data = { v: v_command, w: w_command };
    fetch("http://"+server_addr+":"+server_port+"/raw", {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });
}

// for detecting which key is been pressed w,a,s,d
function updateKey(e) {

    e = e || window.event;

    if (e.keyCode == '87') {
        // up (w)
        document.getElementById("upArrow").style.color = "green";
        document.getElementById("direction").textContent = "FORWARD";
        sendVelocity(0.2, 0);
    }
    else if (e.keyCode == '83') {
        // down (s)
        document.getElementById("downArrow").style.color = "green";
        document.getElementById("direction").textContent = "BACKWARD";
        sendVelocity(-0.2, 0);
    }
    else if (e.keyCode == '65') {
        // left (a)
        document.getElementById("leftArrow").style.color = "green";
        document.getElementById("direction").textContent = "LEFT";
        sendVelocity(0, 0.4);
    }
    else if (e.keyCode == '68') {
        // right (d)
        document.getElementById("rightArrow").style.color = "green";
        document.getElementById("direction").textContent = "RIGHT";
        sendVelocity(0, -0.4);
    }
}

// reset the key to the start state 
function resetKey(e) {

    e = e || window.event;

    document.getElementById("upArrow").style.color = "grey";
    document.getElementById("downArrow").style.color = "grey";
    document.getElementById("leftArrow").style.color = "grey";
    document.getElementById("rightArrow").style.color = "grey";
    document.getElementById("direction").textContent = "";
    sendVelocity(0, 0);
}


// update data for every 50ms
function update_data(){
    setInterval(function(){
        // get image from python server
        client();
    }, 50);
}

update_data();
