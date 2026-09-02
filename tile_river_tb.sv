
//a python model generates three text files with inputs, weights and outputs. Main module of design fills ROMs with weight values. Python code is set to generate 10 sets of test inputs. Testbench read this input file and store the input sets into an array. Outputs are also read from text file and displayed as expected outputs along with the generated outputs. A signal 'poll' is also displayed, this signal compares generated and expected outptus and tells if the outputs match. This design generates outputs with some rounding error, you would notice that in the displayed outputs, comparing generated and model's outputs(expected outputs). 

module tile_river_tb();
	
	logic clk, rst;
	logic [15:0] testinput [1:10];
	logic testinput_valid, testoutput_ready;
	wire testinput_ready, testoutput_valid;
	wire [15:0] testoutput1, testoutput2;
	logic [15:0] testset [1:100];
	logic [15:0] expout[1:20];
	logic [15:0] expout1,expout2;
	logic [3:0] count;
	logic poll;

	tile_river tile_river_inst(clk, rst, testinput, testinput_valid, testoutput_ready, testinput_ready, testoutput_valid, testoutput1, testoutput2);


	initial clk=0;
	always #5 clk = ~clk;

	initial rst = 0;


	initial begin
		$readmemh("inputs.txt", testset);
		$readmemh("outputs.txt", expout);
	end



	initial begin

		@(posedge clk);
		#1;
		rst = 1; testinput_valid = 0; testoutput_ready = 0;

		

		@(posedge clk);
		count<=0;
		


		@(posedge clk);
		#1;testinput_valid = 1; rst = 0;
		for (int i=1;i<=10;i++) begin
			testinput[i] = testset[count*10+i]; 
		end


		for(int i=0; i<5;i++)
			@(posedge clk);


		@(posedge clk);
		#1;
		testoutput_ready = 1;


	end

	always @(*) begin

	end


	always @(posedge clk) begin
		if (testoutput_ready && testoutput_valid) begin
			if (count<11) begin
				count<=count+1;
				expout1 = expout[count*2+1];
				expout2 = expout[count*2+2];
	        	$display("output1 = %h   expout1 = %h   poll=%b" ,testoutput1,expout1,testoutput1==expout1);
	        	$display("output2 = %h   expout2 = %h   poll=%b" ,testoutput2,expout2,testoutput2==expout2);
	        	$display("\n");
	   		end
	   	end
   	end


	always @(*) begin
		for (int i=1;i<=10;i++) begin
			testinput[i]=testset[count*10+i]; 
		end
	end


	initial begin
		wait (count==10);
		$display("System processed %d sets of inputs", count);
		#1;
		$finish;

	end



endmodule














