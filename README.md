# Tile-River
Implementing data propagation network across fully connected neural network layers

This is a Neural Network (NN) structure using a tile structure (in this every tile acts as a node for a NN, see the attached image). This design uses a a 10x10 tile structure. Here data flows from some input nodes all the way  to the output nodes (in other words, building a network communication implementation). Half Precision Floating point numbers are used(all Normalized numbers).

1- We used the tile architecture(see attached image in the repo) and mapped a NN to the tile architecture. The NN has 10 input nodes, 8 and 6 nodes on the 2nd and the 3rd hidden layers respectively  and 2 output nodes on the output layer. 
2- For the input layer, assume some fixed random values  inputs to this layer  between 1 and -1  
3- Assume all weights are set  randomly between 0 and 1
4- It is a fully connected NN (that is every node on the 10 input layer is connected to every 8 nodes on the first hidden layer, and so on)
5- Every nodes sums up the weights times the corresponding outputs from previous layers
6- Information is passed to the next node serially (on a single line)  in a packet type  format as shown here: {0x7E, information, 0x7E} The hex values 0x7E indicate the start and end of a packet.
When transmitting a information data byte 0x7E, it is expanded to two bytes 0x7D, 0x5E
When transmitting a information data byte 0x7D, it is expanded to two bytes 0x7D, 0x5D
During a reception, if the logic encounters 0x7D, it will remove it from the stream and instead XOR 0x20 with the next byte to get the actual transmitted information data of either 0x7E or 0x7D
7- Every node needs to receive all inputs before generating a corresponding output
8- Wrote a simple behavioral model in Python for this design, to verify output node values of SystemVerilog model match the behavioral model outputs


Some explanation regarding the tiles:
1- Every tile has the same structure
2- Tiles on the boundary access the outside (so you can choose the boundary tiles as your primary input and output nodes)
3- Tiles on the corner can access only 2 neighbors (as shown in the attached image)
4- Tiles on the boundary, other than the corner ones, can access 3 neighbors (as shown in the attached image)
5- All other tiles can access 4 neighbors (as shown in the attached image)




