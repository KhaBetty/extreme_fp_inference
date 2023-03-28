`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 29.11.2018 11:46:33
// Design Name: 
// Module Name: normalise
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

module normalized(mxy1,s,s1,s2,s3,clk,ex,sr,exy,mxy);
input[2:0]mxy1;
input s,s1,s2,s3,clk;
input[2:0]ex;
output reg sr;
output reg[2:0]exy;
output reg[1:0]mxy;
reg [2:0]mxy2;
always@(posedge clk)
begin
sr=s?s1^(mxy1[2]&s3):s2^(mxy1[2]&s3);
mxy2=(mxy1[2]&s3)?~mxy1+3'b1:mxy1;
mxy=mxy2[2:1];
exy=ex;
repeat(2)
begin
if(mxy[1]==1'b0)
begin
mxy=mxy<<1'b1;
exy=exy-3'b1;
end
end
end
endmodule
