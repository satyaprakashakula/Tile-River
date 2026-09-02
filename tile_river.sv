
/* Nodes mapped to tiles

Nodes are mapped to 10x10 tile network in the below fashion. 10 Input layer nodes are mapped to tiles 1-10. 8 HiddenLayer1 nodes to tiles 13-20, 6 hiddenLayer2 to 23-28, and 2 output nodes to tiles 30 and 40. Tile no.29 is included for communication from HiddenLayer2 to output nodes. Data flows or hops across tiles in a river or snake like fashion from the first tile(or node) of input layer{tile no.1} to the last node of output layer(tile no.40). For example, during input lyer to hidden layer1, input values hop from tile no.1 to tile no.13 in a sequential fashion in the direction/order shown below in the image. So, input value at tile1 hops to tile2, next to tile3,....to tile10, then to next in the sequence ie., tile20, from tile20 to tile19,....till value reaches tile13. Such hops happen untill all input values reach tile13. Below code is structured into 3 modules, a module that implements a tile, which is instantiated in the main module to create the tile network. And a float_real_float module to deal with half precision floats and 'real' conversions.  All modules are in this file.



1  -->  2  -->   3  -->   4  -->   5  -->  6  -->   7  -->   8  -->   9  -->   10
																				|
																				|
																				|
*       *       13  <--  14  <--  15  <--  16  <--  17  <--  18  <--  19  <--  20
				|
				|
				|
*       *       23  -->  24  -->  25  -->  26  -->  27  -->  28  --> (29)  -->  30
																				|
																				|
																				|
*       *       *        *        *        *         *        *        *       40




*/



//this module is implementation of a single tile or node. This tile is instantiated to create the tile network. Each tile in the network at all times is involved in 2 tasks, either it is hopping data(ie., transferring and receiving data from neighboring tiles) or computing(ie., generating layer outputs). In the module an FSM is implemeted to determine when each tile would either be hopping or computing or just not doing either, and also the to-do logic when a hop or compute of a node is set

module tile(loc, clk, rst, data_in, valid_in, data_out, valid_out, testinput, testinput_valid, testinput_ready,  testoutput_valid, testoutput_ready, weight, testoutput);


	//each tile module receives a value called 'loc' from the main module, this tells each tile what is its location on the tile network. testinput is input array with 10 test inputs to the neural network. testinput_valid and testoutput_ready are input signals used for handshaking, to send and receive inputs/outputs. Each tile has a register called reg_nodeval, this register is used both to store the initial inputs by the input layer tiles, and then also used by the other layer tiles to store MAC operation value(basically output of the layer). Each tile has on output called testoupt, value in reg_nodeval is assigned to this. The main module sends the testoutput of output nodes(tile number 30 and 40) as the output of the neural network layers. Each tile generates a valid_out signal to transmit data packet to its neighboring tile, and data_out is the 1-bit data that is serially transmitted over each clock cyle. Each tiles data_out and valid_out are connected to the data_in and valid_out of the next tile. 'mem' and 'weight' store weights, and inputs from previous layer. counter register here is to help with the output computation cycles. m,w, acc_fp16 are operands in the MAC operation(basically memory value, weight value, and sum). reg_tx, reg_rs store the data packet being transmitted/received, and counters count_tx & count_rx help to keep track of the packet flow. packet_size is the size of data packet. Each tile has signals called hop and compute, that tell them when to transmit/receive data and when to generate layer outputs.These signals are set and sent by the FSM in the tile, since the tile module is instantiated to replicate all the tiles in the tile network, each tile has the same FSM running across all of them. variable cycle_count is used to keep track of the hop cycle. 
	
	input clk, rst;
	input [6:0] loc;
	input [15:0] testinput[1:10];
	input testinput_valid, testoutput_ready;
	output wire testinput_ready, testoutput_valid;

	logic [15:0] reg_nodeval;

	input  data_in, valid_in;
	output wire data_out;
	output wire valid_out;

	logic [15:0] mem [0:15]; 
	logic [3:0] counter;
	input [15:0] weight [0:9]; //while creating weight array, make sure in-memory locations are from 0 to 9 and not 1 to 10

	logic [63:0] reg_tx,reg_rx, reg_rx_reverse;
	logic [5:0] packet_size, count_tx, count_rx;

	output wire [15:0] testoutput;

	logic [15:0] m, w;
	logic [15:0] acc_fp16;

	logic [3:0] offset;


	//fsm states, and arrays called phase lists that store the sequence of nodes. Phase1 refers to the stage of transfer of data from input layer to hidden layer1 tiles. Array store the nodes in the sequence in which data hops across the tiles and also the position of each tile's location in the sequence is stored in the array. Variable cycle count keeps track of the data packet transfer between two tiles and also the computation cycles when a node is generating output

	enum {START, PHASE1, PHASE1_COMPUTE, PHASE2, PHASE2_COMPUTE, PHASE3, PHASE3_COMPUTE, RESULTS} state, next;
	logic [6:0] phase1_list [1:18][1:2] = '{'{1,1}, '{2,2}, '{3,3}, '{4,4}, '{5,5}, '{6,6}, '{7,7}, '{8,8}, '{9,9}, '{10,10}, '{20,11}, '{19,12}, '{18,13}, '{17,14}, '{16,15}, '{15,16}, '{14,17}, '{13,18}};
	logic [4:0] iteration;
	logic [100:0] hop, compute;
	logic [5:0] cycle_count;
	logic [6:0] phase2_list [1:14][1:2] = '{'{20,1}, '{19,2}, '{18,3}, '{17,4}, '{16,5}, '{15,6}, '{14,7}, '{13,8}, '{23,9}, '{24,10}, '{25,11}, '{26,12}, '{27,13}, '{28,14}};
	logic [6:0] phase3_list [1:9][1:2] = '{'{23,1}, '{24,2}, '{25,3}, '{26,4}, '{27,5}, '{28,6}, '{29,7}, '{30,8}, '{40,9}};



	//below is logic for what a tile does when it is hop time and when it's compute time for the tile. Each tile has a hop signal and a compute signal to tell it what to do at at any time. An FSM determines the flow of the neural network system, it determines when each tile would be transferring/receiving data across or when would a tile be doing computation(generating layer outputs), the fsm generates hop and compute signals for each tile. Below logic is for what is done when hop/compute signals are raised for a node. During a hop, a node simultaneously sends its data to its next neighbor in the flow chain and receives data from its preceding tile. When hop==1, a tile first converts its data to be transmitted into a packet, then sets its data_out signals as valid. And over next few clock cycles, the tile sends the packet, and accepts data from preceding tile based on the data_in to it being set to valid{data_out of a tile is connected to the data_in of the next tile}. Once the entire packet is sent/received, the tile decodes the data packet, and stores them into a memory for later use in MAC computations. When a compute signal is set for a tile, values in 'memory' and another ROM memory called 'weight' are retrieved to compute outputs of the layer. 
	
	always_ff @(posedge clk) begin


	//registers like reg_tx,reg_rx are used hold the data packet that being sent and received. 

		if (rst) begin
			counter<=0;
			reg_tx<=0;
			//reg_rx_reverse<=0; 
			reg_rx<=0;
			packet_size<=0;
			count_tx<=0;
			count_rx<=0;
			reg_nodeval<=0;
			for (int i=0;i<16;i++) begin
				mem[i]<=0;
			end
		end


		else begin

			//resetting values, to be ready for hop and compute
			if (state==START) begin   
				counter<=0;
				reg_tx<=0;
				reg_rx_reverse<=0; 
				reg_rx<=0;
				count_tx<=0;
				count_rx<=0;
				for (int i=0;i<16;i++) begin
					mem[i]<=0;
				end

				if(state==START && testinput_ready==1 && testinput_valid==1 && loc>=1 && loc<=10) begin  //storing inputs received into reg_nodeval 
					reg_nodeval<=testinput[loc];
				end

			end


			//when its hop time
			if (hop[loc]==1) begin

				//cycle_count<1 refers to initially, data to be sent, being encoded to packets
				if (cycle_count<1) begin

					//to transmit all the inputs or values from one layer of nodes to the next, certain number of hops are needed to be perfomed by the tiles. And you don't have to packetize the data for each hop, here for the first hop data is packetized to be sent to nieghboring tiles, and then for the next hop, the tiles can just pass on the packet data forward, no need to again convert them to packets.

					if (iteration==0) begin

						if ((reg_nodeval[7:0]==8'h7E || reg_nodeval[7:0]==8'h7D) && (reg_nodeval[15:8]==8'h7E || reg_nodeval[15:8]==8'h7D)) begin

							packet_size<=48;

							if (reg_nodeval[7:0]==8'h7E && reg_nodeval[15:8]==8'h7E)
								
								reg_tx <= {8'h7E,8'h7D,8'h5E,8'h7D,8'h5E,8'h7E};

							if (reg_nodeval[7:0]==8'h7D && reg_nodeval[15:8]==8'h7D)

								reg_tx <= {8'h7E,8'h7D,8'h5D,8'h7D,8'h5D,8'h7E};

							if (reg_nodeval[7:0]==8'h7D && reg_nodeval[15:8]==8'h7E)

								reg_tx <= {8'h7E,8'h7D,8'h5E,8'h7D,8'h5D,8'h7E};

							if (reg_nodeval[7:0]==8'h7E && reg_nodeval[15:8]==8'h7D)

								reg_tx <= {8'h7E,8'h7D,8'h5D,8'h7D,8'h5E,8'h7E};
						end


						else if ((reg_nodeval[7:0]==8'h7E || reg_nodeval[7:0]==8'h7D) && (reg_nodeval[15:8]!=8'h7E || reg_nodeval[15:0]!=8'h7D)) begin

							packet_size<=40;

							if (reg_nodeval[7:0]==8'h7E)

								reg_tx <= {8'd0,8'h7E,reg_nodeval[15:8],8'h7D,8'h5E,8'h7E};

							if (reg_nodeval[7:0]==8'h7D)

								reg_tx <= {8'd0,8'h7E,reg_nodeval[15:8],8'h7D,8'h5D,8'h7E};

						end

						else if ((reg_nodeval[15:8]==8'h7E || reg_nodeval[15:8]==8'h7D) && (reg_nodeval[7:0]!=8'h7E || reg_nodeval[7:0]!=8'h7D)) begin

							packet_size<=40;

							if (reg_nodeval[15:8]==8'h7E)

								reg_tx <= {8'd0,8'h7E,8'h7D,8'h5E,reg_nodeval[7:0],8'h7E};

							if (reg_nodeval[15:8]==8'h7D)

								reg_tx <= {8'd0,8'h7E,8'h7D,8'h5D,reg_nodeval[7:0],8'h7E};

						end

						else begin
							packet_size<=32;
							reg_tx <= {16'd0,8'h7E,reg_nodeval,8'h7E};  
						end

					end


					
					else begin

						reg_nodeval<=0;
						reg_tx<=reg_rx;     //for the next transmissions/hops, just use the received encoded packet to transmit
						reg_rx_reverse<=0; 
						reg_rx<=0;
						count_tx<=0;
						count_rx<=0;
					end

				end


				//during hop (here until both the tramsission and reception done)
				if (cycle_count>0 && cycle_count<49) begin

					//valid_out<=(cycle_count<packet_size)?1:0;

					//at transmitter of a tile
					if (valid_out==1)  //make valid_out high for it's transmitted packet_size number of cycles
						count_tx<=count_tx+1;

					//at receiver of a tile
					if (valid_in==1)  begin
						reg_rx[count_rx]<=data_in;   
						count_rx<=count_rx+1;
					end

				end


				//after hop  --> decode the received packet and store it into memory
				if (cycle_count==49) begin

					counter<=counter+1;
					packet_size<=count_rx;  

					if (count_rx==32)
						mem[offset-counter]<=reg_rx[23:8];
					if (count_rx==40) begin
						if (reg_rx[31:24]=='h7D) begin
							if(reg_rx[23:16]=='h5E)
								mem[offset-counter]<={8'h7E,reg_rx[15:8]};
							else
								mem[offset-counter]<={8'h7D,reg_rx[15:8]};
						end
						if (reg_rx[23:16]=='h7D) begin
							if(reg_rx[15:8]=='h5E)
								mem[offset-counter]<={reg_rx[31:24], 8'h7E};
							else
								mem[offset-counter]<={reg_rx[31:24], 8'h7D};
						end
					end
					if (count_rx==48) begin
						if(reg_rx[31:24]==8'h5E && reg_rx[15:8]==8'h5E)
							mem[offset-counter]<={8'h7E, 8'h7E};
						if(reg_rx[31:24]==8'h5D && reg_rx[15:8]==8'h5D)
							mem[offset-counter]<={8'h7D, 8'h7D};
						if(reg_rx[31:24]==8'h5E && reg_rx[15:8]==8'h5D)
							mem[offset-counter]<={8'h7E, 8'h7D};
						if(reg_rx[31:24]==8'h5D && reg_rx[15:8]==8'h5E)
							mem[offset-counter]<={8'h7D, 8'h7E};
					end

				end			
			end



			//after the transfer from a layer to layer is done, when the layer2 tiles receive compute signal to generate outputs
			if (compute[loc]==1) begin

				//do until all weights and memory value products are accumulated
				if ((state==PHASE1_COMPUTE && cycle_count<10) || (state==PHASE2_COMPUTE && cycle_count<8) || (state==PHASE3_COMPUTE && cycle_count<6)) begin
						reg_nodeval<= acc_fp16; //reg_nodeval holds the accumulated value. The 16-bit memory,weight and reg_nodeval values are sent to a module which decodes these 16-bit values into 'real' numbers, does MAC operation, check for saturation and sub-normal constraints on the accumulated value, and then sends the half preceision floating notation of the 'real' accumulated value. acc_fp16 is that value.
				end

				

				reg_rx_reverse<=0; //set these registers to value 0, else they would misrepresent the value that will be received during the hop
				reg_rx<=0;
				count_tx<=0;
				count_rx<=0;

			end

		end
	end


	always_comb begin
		if (rst||state==START)
			offset=0;
		if (state==PHASE1)
			offset=9;
		if (state==PHASE2)
			offset=7;
		if (state==PHASE3)
			offset=5;
	end

	assign data_out = reg_tx[count_tx]; //here, data_out is of current tile and reg_tx is also of current tile
	assign testoutput = reg_nodeval;
	assign m = mem[cycle_count];
	assign w = weight[cycle_count];

	float_real_float float_real_float_inst(m,w,reg_nodeval,acc_fp16);   //calling or instantiating the float_real_float module that does MAC operation on the half precision floating point operands sent to it
	

	//The below always_ff and always_comb blocks correspond to the FSM, that determines the timing and data flow behavior across the neural network nodes and layers. There are 7 states in total(START, PHASE1, PHASE1_COMPUTE, PHASE2, PHASE2_COMPUTE, PHASE3, PHASE3_COMPUTE, RESULTS). Which isn't done here, but ideally, except START state all other states could be generalized into two states, one where hopping happens and the other where computation happens. On a reset, system starts with START state- here inputs to the neural network are received, when there is a handshake(testinput_ready==testinput_valid). States are designed like this - a timeperiod or phase to transmit all the data from one layer to its next layer, and then another phase or stage or timeperiod during which the layer that received values, generates the outputs. These two stages are names phase and phase-compute. So, in here, six states are required to transmit data from input nodes layer to the output nodes layer generating outputs. In PHASE1 state - input nodes transfer inputs received to hiddenlayer1. In PHASE1_COMPUTE state hiddenlayer1 nodes generate outputs. Likewise, PHASE2, PHASE2_COMPUTE, PHASE3, PHASE3_COMPUTE for hiddenlayer1->hiddenlayer2-outputlayer flow. And RESULTS state to pass the outputs to the testbench on a handshake(testoutput_valid==testoutput_ready). State movement is done based on 'cycle_count' and 'teration'. A 'hop' is the transfer or reception of full data packet from one tile to the other. As mentioned earlier, it takes some cycles for a data packet hop. Variable 'cycle_count' helps keep track of that, to determine if the hop is done. To send all the values from one layer to the next, certains number of hops are perfomred. For example to send all the inputs from input layer to hidden layer1, 17 hops are required. Variable 'iteration' helps keep track of the hop count. 


	//this always_ff block is for updating 'cycle_count' and 'iteration'

	always_ff @(posedge clk) begin
		if (rst) begin
			state<=START;
			iteration<=0;
			cycle_count<=0;
		end

		else
			state<=next;

		if(state==PHASE1) begin
			if (iteration<17)
				cycle_count<=(cycle_count==49)?0:cycle_count+1;
			if (cycle_count==49)
				iteration<=iteration+1;
		end

		if (state==PHASE1_COMPUTE) begin
			iteration<=0;
			cycle_count<=(cycle_count==10)?0:cycle_count+1;
		end

		if (state==PHASE2) begin
			if (iteration<13)
				cycle_count<=(cycle_count==49)?0:cycle_count+1;
			if (cycle_count==49)
				iteration<=iteration+1;
		end

		if (state==PHASE2_COMPUTE) begin
			iteration<=0;
			cycle_count<=(cycle_count==8)?0:cycle_count+1;
		end

		if (state==PHASE3) begin
			if (iteration<8)
				cycle_count<=(cycle_count==49)?0:cycle_count+1;
			if (cycle_count==49)
				iteration<=iteration+1;
		end

		if (state==PHASE3_COMPUTE) begin
			iteration<=0;
			cycle_count<=(cycle_count==6)?0:cycle_count+1;
		end


	end


	//this always_comb block is for setting 'hop' and 'compute' for tiles, and determining next_state. In inputlayer->hiddenlayer1 communication, input values flow from the inputs to the next layer in a snake-like fashion. As in each data hops data to its successcive neighbor. 10 input layer nodes are mapped to tiles 1 to 10. 8 layer2 nodes are mapped to tile numbers 13 to 20. Here, data flows or hops in this way forward, untill each tile in Layer2 receives all the 10 inputs: 1=>2=>3=>4=>5=>5=>6=>7=>8=>9=>10>20=>19=>18=>17=>16=>15=>14=>13.  So input values at tile 1 hops to tile2, and the input value at tile2 hops to tile3, so, during the first iteration of hop - tiles 1 to 10, simultaneously transmit their data to the the next in line tile. So, at the end of first iteration of hop, tile20 has tile10's input value, and tile10 has tile9's value and such. Now, for the second iteration of hop, tile1 doesn't participate in hopping, as it already has transmitted its input value to tile2. After 17 such iterations of hopping, the last tile in the order, tile no.13 will have all the 10 input values and that is when we end with state PHASE1. In state PHASE1_COMPUTE, Layer2 generates outputs. In PHASE2 Layers outputs are transmitted to Layers over 13 iterations of hops. And so on until output layer generates outputs. From RESULTS state, loop to START state to receive next set of inputs. 
	
	always_comb begin
		
		for (int i=0; i<101; i++) begin
				hop[i]=0;
				compute[i]=0;
		end


		if (state==START) begin
			next = START;
		
			//testinput_ready=1;

			if (testinput_valid==1) 
				next = PHASE1; 
		end

		if (state==PHASE1) begin
			next = PHASE1;
			if (iteration<17) begin
				for (int i=1;i<11;i++) begin
					hop[phase1_list[i][1]] = ((iteration+1==phase1_list[i][2]) || (iteration+1<phase1_list[i][2])) ? 1 : 0; //In each PHASE of hopping, not all the tiles participate in all the ieterations of hops. For example in PHASE1, tile no.1 only hops value during first iteration of hopping, and say, tile no.10 participates in hopping from iteration 1 to iteration 11, the time by which it would have all the 10 input values with it. And say, tile no.13, would be active from iteration 8 to iteration 17, beacuse during iteration8 of hopping it receives the first input value hopped to it by its preceding tile tile14, and after iteration17 it would receive all 10 input values. This line of logic, determines when the 'hop' signal of a tile should be set based on the tile's location in the sequence of data flow chain. The order of tiles in the sequence is stored in the pahse list arrays. phase1_list corresponds to the data flow from input layer to hidden layer1, similarly pahse2_list and phase3_list.
				end
				for (int i=11;i<19;i++) begin
					hop[phase1_list[i][1]] = ((iteration+1>phase1_list[i][2]-10-1) && (iteration+1<phase1_list[i][2]+1)) ? 1 : 0;
				end
			end

			if (iteration==17)
				next = PHASE1_COMPUTE;
		end

		if (state==PHASE1_COMPUTE) begin
			next = PHASE1_COMPUTE;
			if (cycle_count<10) begin
				for (int i=11; i<19; i++) begin
					compute[phase1_list[i][1]] = 1;
				end
			end
			if (cycle_count==10)
				next = PHASE2;
		end

		if (state==PHASE2) begin
			next = PHASE2;
			if (iteration<13) begin
				for (int i=1;i<9;i++) begin
					hop[phase2_list[i][1]] = ((iteration+1==phase2_list[i][2]) || (iteration+1<phase2_list[i][2])) ? 1 : 0;
				end
				for (int i=9;i<15;i++) begin
					hop[phase2_list[i][1]] = ((iteration+1>phase2_list[i][2]-8-1) && (iteration+1<phase2_list[i][2]+1)) ? 1 : 0;
				end
			end

			if (iteration==13)
				next = PHASE2_COMPUTE;
		end

		if (state==PHASE2_COMPUTE) begin
			next = PHASE2_COMPUTE;
			if (cycle_count<8) begin
				for (int i=9; i<15; i++) begin
					compute[phase2_list[i][1]] = 1;
				end
			end
			if (cycle_count==8)
				next = PHASE3;
		end

		if (state==PHASE3) begin
			next = PHASE3;
			if (iteration<8) begin
				for (int i=1;i<7;i++) begin
					hop[phase3_list[i][1]] = ((iteration+1==phase3_list[i][2]) || (iteration+1<phase3_list[i][2])) ? 1 : 0;
				end
				for (int i=7;i<10;i++) begin
					hop[phase3_list[i][1]] = ((iteration+1>phase3_list[i][2]-6-1) && (iteration+1<phase3_list[i][2]+1)) ? 1 : 0;
				end
			end

			if (iteration==8)
				next = PHASE3_COMPUTE;
		end

		if (state==PHASE3_COMPUTE) begin
			next = PHASE3_COMPUTE;
			if (cycle_count<6) begin
				for (int i=8; i<10; i++) begin
					compute[phase3_list[i][1]] = 1;
				end
			end
			if (cycle_count==6)
				next = RESULTS;
		end

		if (state==RESULTS) begin
			next = RESULTS;
			if (testoutput_ready==1)
				next = START;
		end


	end

	
	assign testoutput_valid = (state==RESULTS);
	assign testinput_ready = (state==START);
	assign valid_out = (hop[loc]==1 && cycle_count>0 && cycle_count<=packet_size);


endmodule



//This is the main module that instantiates all the tiles required for this neural network. Also, for weights, ROMs called 'weight' are created and passed to each tile. These ROMs are filled with values from a text file(of weights) generated by the python model.

module tile_river(clk, rst, testinput, testinput_valid, testoutput_ready, input_ready, output_valid, testoutput1, testoutput2);
	
	input clk, rst;
	logic [6:0] loc[1:100];
	input [15:0] testinput[1:10];
	input testinput_valid, testoutput_ready;
	wire  [0:0] testinput_ready [1:100];
	wire  [0:0] testoutput_valid [1:100];
	output wire input_ready, output_valid;

	output wire [15:0] testoutput1, testoutput2;
	wire [15:0] testoutput [1:100];


	logic  data_in[1:100], valid_in[1:100];
	wire   data_out [0:100];
	wire  valid_out [0:100];

	logic [15:0] weight [1:100][0:9]; //while creating weight array, make sure in-memory locations are from 0 to 9 and not 1 to 10
	logic [15:0] mass [1:140];


	assign input_ready=testinput_ready[1];
	assign output_valid=testoutput_valid[30];

	assign testoutput1=testoutput[30];
	assign testoutput2=testoutput[40];

	logic [6:0] node_list [1:27] = '{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 19, 18, 17, 16, 15, 14, 13, 23, 24, 25, 26, 27, 28, 29, 30, 40};
	logic [6:0] weight_list [1:16] = '{20, 19, 18, 17, 16, 15, 14, 13, 23, 24, 25, 26, 27, 28, 30, 40};


	/*always_ff @(posedge clk) begin
		for (int i=1; i<=100; i++) begin
			for (int j=0; j<10; j++) begin
				weight[i][j]<=16'b0011100000000000; //is 0.5   //randomize ($urandom, $urandom_range(1,10))
			end
		end
	end */

	initial begin
		$readmemh("weights.txt",mass);
		for (int i=1;i<=16;i++) begin
			for (int j=0;j<=9;j++) begin
				if (i<9)
					weight[weight_list[i]][j] = mass[(i-1)*10+j+1];
			end
		end

		for (int i=1;i<=16;i++) begin
			for (int m=0;m<=7;m++) begin
				if (i>8 && i<15)
					weight[weight_list[i]][m] = mass[80+(i-8-1)*8+(m+1)];
			end
		end

		for (int i=1;i<=16;i++) begin
			for (int n=0;n<=5;n++) begin
				if (i>14)
					weight[weight_list[i]][n] = mass[128+(i-14-1)*6+(n+1)];
			end
		end
	end



	always_comb begin
		for(int i=1;i<=100;i++) begin
			loc[i] = i;
		end
	end

	

	generate
		genvar i;
		for (i=1; i<=10; i++) begin
			//if (node_list[i]<=100)
				tile firstlayer(loc[i], clk, rst, data_out[i-1], valid_out[i-1], data_out[i], valid_out[i], testinput, testinput_valid, testinput_ready[i], testoutput_valid[i], testoutput_ready, weight[i], testoutput[i]);
		end
	endgenerate

	generate
		genvar j;
		for (j=13; j<=20; j++) begin
				tile secondlayer(loc[j], clk, rst, data_out[j==20?10:j+1], valid_out[j==20?10:j+1], data_out[j], valid_out[j], testinput, testinput_valid, testinput_ready[j], testoutput_valid[j], testoutput_ready, weight[j], testoutput[j]);
		end
	endgenerate

	generate
		genvar k;
		for (k=23; k<=28; k++) begin
				tile thirdlayer(loc[k], clk, rst, data_out[k==23?13:k-1], valid_out[k==23?13:k-1], data_out[k], valid_out[k], testinput, testinput_valid, testinput_ready[k], testoutput_valid[k], testoutput_ready, weight[k], testoutput[k]);
		end
	endgenerate

	tile tile_29(7'd29, clk, rst, data_out[28], valid_out[28], data_out[29], valid_out[29], testinput, testinput_valid, testinput_ready[29], testoutput_valid[29], testoutput_ready, weight[29], testoutput[29]);

	tile tile_30(7'd30, clk, rst, data_out[29], valid_out[29], data_out[30], valid_out[30], testinput, testinput_valid, testinput_ready[30], testoutput_valid[30], testoutput_ready, weight[30], testoutput[30]);

	tile tile_40(7'd40, clk, rst, data_out[30], valid_out[30], data_out[40], valid_out[40], testinput, testinput_valid, testinput_ready[40], testoutput_valid[40], testoutput_ready, weight[40], testoutput[40]);


endmodule



//this moudle is for halfpercisionfloat - to - MAC operation - to - real - and back to - fp16 conversion. This module receives 16-bit MAC operation operands, decodes them to 'real' values, does MAC operation. And this value is then checked if the value underflows half precision float's normalized values and range overflows, and the value is saturated to limits if overflow/underflow. Now, this MAC value is encoded to 16bit fp16 notation, and sent as output.

module float_real_float (
    input  logic [15:0] a,
    input  logic [15:0] b,
    input  logic [15:0] sum,
    output logic [15:0] acc_fp16
);

    real a_real, b_real, sum_real, acc;

    localparam real MAX_FP16     = 65504.0;
    localparam real MIN_NORM_POS = 2.0**-14;

    function automatic real fp16_to_real(logic [15:0] fp);
        logic sign;
        int   exp;
        real  fraction;

        if (fp[14:10] == 0) begin
            return 0.0;
        end

        sign     = fp[15];
        exp      = fp[14:10] - 15; 
        fraction = 1.0 + (fp[9:0] / 1024.0);

        return (sign ? -1.0 : 1.0) * fraction * (2.0**exp);
    endfunction

    function automatic logic [15:0] real_to_fp16(real val);
        logic sign;
        real  v;
        int   exp;
        real  frac;
        real  frac_full;
        real  delta;
        int   frac_int;
        int   exp_biased;

        if (val == 0.0) begin
            return 16'b0;
        end

        sign = (val < 0);
        v    = sign ? -val : val;

        exp  = 0;
        frac = v;
        while (frac >= 2.0) begin
            frac = frac / 2.0;
            exp  = exp + 1;
        end
        while (frac < 1.0) begin
            frac = frac * 2.0;
            exp  = exp - 1;
        end

        frac_full = (frac - 1.0) * 1024.0;
        frac_int  = $rtoi(frac_full);

        delta = frac_full - frac_int;
        if (delta > 0.5) begin
            frac_int = frac_int + 1;
        end else if (delta == 0.5) begin
            if (frac_int[0] == 1) begin
                frac_int = frac_int + 1;
            end
        end

        if (frac_int == 1024) begin
            frac_int = 0;
            exp      = exp + 1;
        end

        exp_biased = exp + 15;

        if (exp_biased >= 31) begin
            exp_biased = 30;
            frac_int   = 10'h3FF;
        end

        return {sign, exp_biased[4:0], frac_int[9:0]};
    endfunction

    always_comb begin
        a_real   = fp16_to_real(a);
        b_real   = fp16_to_real(b);
        sum_real = fp16_to_real(sum);

        acc = a_real * b_real + sum_real;

        if      (acc >  MAX_FP16)     acc =  MAX_FP16;
        else if (acc < -MAX_FP16)     acc = -MAX_FP16;
        else if (acc > 0.0 && acc < MIN_NORM_POS)  acc = MIN_NORM_POS;
        else if (acc < 0.0 && acc > -MIN_NORM_POS) acc = -MIN_NORM_POS;

        acc_fp16 = real_to_fp16(acc);
    end
endmodule









