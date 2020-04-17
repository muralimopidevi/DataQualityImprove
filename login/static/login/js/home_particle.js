
function getRandomInt(min, max) {
  return Math.floor(Math.random() * (max - min)) + min;
}

function particlesInit() {
	var generator = document.getElementById("particleGenerator");
	var particleCount = 300;
	for (var i = 0; i < particleCount; i++) {
		var size = getRandomInt(3, 9);
		var n = '<div class="particle" style="top:' + getRandomInt(15, 95) + '%; left:' + getRandomInt(5,95) + '%; width:'
		+ size + 'px; height:' + size + 'px; animation-delay:' + (getRandomInt(0,30)/10) + 's; background-color:rgba('
		+ getRandomInt(80, 160) + ',' + getRandomInt(185, 255) + ',' + getRandomInt(160, 255) + ',' + (getRandomInt(2, 8)/10) + ');"></div>';
		console.log("Particle " + i + ": " + n);
		var node = document.createElement("div");
		node.innerHTML = n;
		generator.appendChild(node);
	}
}

particlesInit();