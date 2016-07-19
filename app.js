var express = require('express');
var app = express();
var http = require('http').Server(app);
var io = require('socket.io')(http);
var fs = require('fs');
var PythonShell = require('python-shell');
var pyshell = new PythonShell('cli.py');
var say = require('say');


pyshell.send('9\n2\n');


pyshell.on('message', function (message) {
  console.log(message);
  	io.emit('event', message);

});
     


var port = 3000;


app.use('/', express.static('public'));

io.on('connection', function(socket){
	
	socket.on('play', function(msg){
		say.speak(msg);
	});
	
});






http.listen(port, function(){
  console.log('listening on *:'+port);
});


