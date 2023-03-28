`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 29.11.2018 16:55:48
// Design Name: 
// Module Name: fpmul
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module fpmul(clk,reset,a,b,fprod);
parameter w=5;
input [w-1:0] a,b;
input clk,reset;

output  [w-1:0] fprod;

wire [w-1:0]s1,s2;
wire [w-1-3:0] m1,m2;
wire [w-1-1:0] routm;
wire [w-1-2:0] e1,e2,oe;
wire signa,signb,carry;

wire [7:0] prod1;
reg [2:0] oute;
reg signout;
reg [7:0] prod;
 reg [w-1-1:0]outm;

  assign s1=a;
  assign s2=b;
  assign e1=s1[3:1];
  assign e2=s2[3:1];
  assign m1[1]=1;
  assign m1[0]=s1[0];
  assign m2[1]=1;
  assign m2[0]=s2[0];
  assign signa=s1[4];
  assign signb=s2[4];

smallalu ee(0,clk,reset,e1,e2,oe);
//assign oute=oe+7'b01111111;
//assign signout=(signa^signb);

wallace_tree v1 (carry,prod1,m1,m2);

//wallacemultiplier(carryf,product1,inp1,inp2);

always @ (posedge clk)
begin
if (reset)
begin
outm=4'b0;
oute=3'b0;
signout=1'b0;
prod=8'b0;
end
else if (prod1[7])
begin
prod=prod1>>1;
outm=prod[7:4];
oute=oe+3'b011+3'b001;
signout=(signa^signb);

end
else
begin
outm=prod1[7:4];
oute=oe+3'b011+3'b010;
signout=(signa^signb);

end



end//always

round rr (clk,reset,outm,routm);

assign fprod[4]=signout;
assign fprod[3:1]=oute;
assign fprod[0]=routm[3:2];

endmodule
