# IIS: Intelligent Bartender

The goal of this project is to provide the software for a bartender robot targeting customers with different disabilities or impairments (deaf, dumb people…) that make it difficult to order drinks or food in a bar. A good solution to tackle that issue is to implement intuitive gesture recognition, where users can complete the ordering process with no further complications. 

The aim of the final prototype is to be a small independent module which can be placed in a bar and can assist users throughout the overall process, allowing them to order either alcoholic or non-alcoholic drinks, the meal of the day and their preferred payment method. 

<img src="http://i.imgur.com/EDpSrvI.png" align="center"  width="400px" >

Execution
---------

For running the code (including the cross-validation, recording, testing, etc.), please run 

python cli.py 

and follow the instructions in the terminal.

For running the user interface, please install node dependencies:
```
npm install express
npm install socket.io
npm install python-shell
npm install say
```

And run it using NodeJS:

```
  node  app.js
```
