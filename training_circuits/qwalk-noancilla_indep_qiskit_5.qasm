// Benchmark was created by MQT Bench on 2023-06-29
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: v1.0.0
// Qiskit version: {'qiskit-terra': '0.24.1', 'qiskit-aer': '0.12.0', 'qiskit-ignis': None, 'qiskit-ibmq-provider': '0.20.2', 'qiskit': '0.43.1', 'qiskit-nature': '0.6.2', 'qiskit-finance': '0.3.4', 'qiskit-optimization': '0.5.0', 'qiskit-machine-learning': '0.6.1'}

OPENQASM 2.0;
include "qelib1.inc";
qreg node[4];
qreg coin[1];
creg meas[5];
h node[0];
cu1(pi/2) node[3],node[0];
u2(pi/4,-pi) node[3];
cx node[2],node[3];
u2(0,3*pi/4) node[3];
h coin[0];
cx coin[0],node[3];
u1(pi/4) node[3];
cx node[1],node[3];
u1(-pi/4) node[3];
cx coin[0],node[3];
u1(pi/4) node[3];
cx node[1],node[3];
u2(pi/4,3*pi/4) node[3];
cx node[2],node[3];
u2(0,3*pi/4) node[3];
cu1(-pi/2) node[3],node[0];
u2(pi/4,-pi) node[3];
cx node[2],node[3];
u2(pi/4,3*pi/4) node[3];
cx node[1],node[3];
u1(-pi/4) node[3];
cx coin[0],node[3];
u1(pi/4) node[3];
cx node[1],node[3];
u1(-pi/4) node[3];
cx coin[0],node[3];
cu1(pi/8) coin[0],node[0];
cx coin[0],node[1];
cu1(-pi/8) node[1],node[0];
cx coin[0],node[1];
cu1(pi/8) node[1],node[0];
u2(pi/4,-pi) node[3];
cx node[2],node[3];
cx node[1],node[2];
cu1(-pi/8) node[2],node[0];
cx coin[0],node[2];
cu1(pi/8) node[2],node[0];
cx node[1],node[2];
u2(pi/8,-pi) node[1];
cu1(-pi/8) node[2],node[0];
cx coin[0],node[2];
p(pi/8) coin[0];
cu1(pi/8) node[2],node[0];
p(pi/8) node[2];
cx coin[0],node[2];
p(-pi/8) node[2];
cx coin[0],node[2];
u2(pi/8,3*pi/4) node[3];
cx node[2],node[3];
p(-pi/8) node[3];
cx coin[0],node[3];
p(pi/8) node[3];
cx node[2],node[3];
p(-pi/8) node[3];
cx coin[0],node[3];
cx node[3],node[1];
p(-pi/8) node[1];
cx node[2],node[1];
p(pi/8) node[1];
cx node[3],node[1];
p(-pi/8) node[1];
cx coin[0],node[1];
p(pi/8) node[1];
cx node[3],node[1];
p(-pi/8) node[1];
cx node[2],node[1];
p(pi/8) node[1];
cx node[3],node[1];
p(-pi/8) node[1];
cx coin[0],node[1];
ccx coin[0],node[3],node[2];
cx coin[0],node[3];
x coin[0];
u2(0,0) node[1];
x node[2];
x node[3];
cu1(pi/2) node[3],node[0];
u2(pi/4,-pi) node[3];
cx node[2],node[3];
u2(0,3*pi/4) node[3];
cx coin[0],node[3];
u1(pi/4) node[3];
cx node[1],node[3];
u1(-pi/4) node[3];
cx coin[0],node[3];
u1(pi/4) node[3];
cx node[1],node[3];
u2(pi/4,3*pi/4) node[3];
cx node[2],node[3];
u2(0,3*pi/4) node[3];
cu1(-pi/2) node[3],node[0];
u2(pi/4,-pi) node[3];
cx node[2],node[3];
u2(pi/4,3*pi/4) node[3];
cx node[1],node[3];
u1(-pi/4) node[3];
cx coin[0],node[3];
u1(pi/4) node[3];
cx node[1],node[3];
u1(-pi/4) node[3];
cx coin[0],node[3];
cu1(pi/8) coin[0],node[0];
cx coin[0],node[1];
cu1(-pi/8) node[1],node[0];
cx coin[0],node[1];
cu1(pi/8) node[1],node[0];
u2(pi/4,-pi) node[3];
cx node[2],node[3];
cx node[1],node[2];
cu1(-pi/8) node[2],node[0];
cx coin[0],node[2];
cu1(pi/8) node[2],node[0];
cx node[1],node[2];
u2(pi/8,-pi) node[1];
cu1(-pi/8) node[2],node[0];
cx coin[0],node[2];
p(pi/8) coin[0];
cu1(pi/8) node[2],node[0];
p(pi/8) node[2];
cx coin[0],node[2];
p(-pi/8) node[2];
cx coin[0],node[2];
u2(pi/8,3*pi/4) node[3];
cx node[2],node[3];
p(-pi/8) node[3];
cx coin[0],node[3];
p(pi/8) node[3];
cx node[2],node[3];
p(-pi/8) node[3];
cx coin[0],node[3];
cx node[3],node[1];
p(-pi/8) node[1];
cx node[2],node[1];
p(pi/8) node[1];
cx node[3],node[1];
p(-pi/8) node[1];
cx coin[0],node[1];
p(pi/8) node[1];
cx node[3],node[1];
p(-pi/8) node[1];
cx node[2],node[1];
p(pi/8) node[1];
cx node[3],node[1];
p(-pi/8) node[1];
cx coin[0],node[1];
ccx coin[0],node[3],node[2];
cx coin[0],node[3];
u2(-pi,-pi) coin[0];
u2(0,0) node[1];
x node[2];
x node[3];
cu1(pi/2) node[3],node[0];
u2(pi/4,-pi) node[3];
cx node[2],node[3];
u2(0,3*pi/4) node[3];
cx coin[0],node[3];
u1(pi/4) node[3];
cx node[1],node[3];
u1(-pi/4) node[3];
cx coin[0],node[3];
u1(pi/4) node[3];
cx node[1],node[3];
u2(pi/4,3*pi/4) node[3];
cx node[2],node[3];
u2(0,3*pi/4) node[3];
cu1(-pi/2) node[3],node[0];
u2(pi/4,-pi) node[3];
cx node[2],node[3];
u2(pi/4,3*pi/4) node[3];
cx node[1],node[3];
u1(-pi/4) node[3];
cx coin[0],node[3];
u1(pi/4) node[3];
cx node[1],node[3];
u1(-pi/4) node[3];
cx coin[0],node[3];
cu1(pi/8) coin[0],node[0];
cx coin[0],node[1];
cu1(-pi/8) node[1],node[0];
cx coin[0],node[1];
cu1(pi/8) node[1],node[0];
u2(pi/4,-pi) node[3];
cx node[2],node[3];
cx node[1],node[2];
cu1(-pi/8) node[2],node[0];
cx coin[0],node[2];
cu1(pi/8) node[2],node[0];
cx node[1],node[2];
u2(pi/8,-pi) node[1];
cu1(-pi/8) node[2],node[0];
cx coin[0],node[2];
p(pi/8) coin[0];
cu1(pi/8) node[2],node[0];
p(pi/8) node[2];
cx coin[0],node[2];
p(-pi/8) node[2];
cx coin[0],node[2];
u2(pi/8,3*pi/4) node[3];
cx node[2],node[3];
p(-pi/8) node[3];
cx coin[0],node[3];
p(pi/8) node[3];
cx node[2],node[3];
p(-pi/8) node[3];
cx coin[0],node[3];
cx node[3],node[1];
p(-pi/8) node[1];
cx node[2],node[1];
p(pi/8) node[1];
cx node[3],node[1];
p(-pi/8) node[1];
cx coin[0],node[1];
p(pi/8) node[1];
cx node[3],node[1];
p(-pi/8) node[1];
cx node[2],node[1];
p(pi/8) node[1];
cx node[3],node[1];
p(-pi/8) node[1];
cx coin[0],node[1];
ccx coin[0],node[3],node[2];
cx coin[0],node[3];
x coin[0];
u2(0,0) node[1];
x node[2];
x node[3];
cu1(pi/2) node[3],node[0];
u2(pi/4,-pi) node[3];
cx node[2],node[3];
u2(0,3*pi/4) node[3];
cx coin[0],node[3];
u1(pi/4) node[3];
cx node[1],node[3];
u1(-pi/4) node[3];
cx coin[0],node[3];
u1(pi/4) node[3];
cx node[1],node[3];
u2(pi/4,3*pi/4) node[3];
cx node[2],node[3];
u2(0,3*pi/4) node[3];
cu1(-pi/2) node[3],node[0];
u2(pi/4,-pi) node[3];
cx node[2],node[3];
u2(pi/4,3*pi/4) node[3];
cx node[1],node[3];
u1(-pi/4) node[3];
cx coin[0],node[3];
u1(pi/4) node[3];
cx node[1],node[3];
u1(-pi/4) node[3];
cx coin[0],node[3];
cu1(pi/8) coin[0],node[0];
cx coin[0],node[1];
cu1(-pi/8) node[1],node[0];
cx coin[0],node[1];
cu1(pi/8) node[1],node[0];
u2(pi/4,-pi) node[3];
cx node[2],node[3];
cx node[1],node[2];
cu1(-pi/8) node[2],node[0];
cx coin[0],node[2];
cu1(pi/8) node[2],node[0];
cx node[1],node[2];
u2(pi/8,-pi) node[1];
cu1(-pi/8) node[2],node[0];
cx coin[0],node[2];
p(pi/8) coin[0];
cu1(pi/8) node[2],node[0];
p(pi/8) node[2];
cx coin[0],node[2];
p(-pi/8) node[2];
cx coin[0],node[2];
u2(pi/8,3*pi/4) node[3];
cx node[2],node[3];
p(-pi/8) node[3];
cx coin[0],node[3];
p(pi/8) node[3];
cx node[2],node[3];
p(-pi/8) node[3];
cx coin[0],node[3];
cx node[3],node[1];
p(-pi/8) node[1];
cx node[2],node[1];
p(pi/8) node[1];
cx node[3],node[1];
p(-pi/8) node[1];
cx coin[0],node[1];
p(pi/8) node[1];
cx node[3],node[1];
p(-pi/8) node[1];
cx node[2],node[1];
p(pi/8) node[1];
cx node[3],node[1];
p(-pi/8) node[1];
cx coin[0],node[1];
ccx coin[0],node[3],node[2];
cx coin[0],node[3];
u2(-pi,-pi) coin[0];
u2(0,0) node[1];
x node[2];
x node[3];
cu1(pi/2) node[3],node[0];
u2(pi/4,-pi) node[3];
cx node[2],node[3];
u2(0,3*pi/4) node[3];
cx coin[0],node[3];
u1(pi/4) node[3];
cx node[1],node[3];
u1(-pi/4) node[3];
cx coin[0],node[3];
u1(pi/4) node[3];
cx node[1],node[3];
u2(pi/4,3*pi/4) node[3];
cx node[2],node[3];
u2(0,3*pi/4) node[3];
cu1(-pi/2) node[3],node[0];
u2(pi/4,-pi) node[3];
cx node[2],node[3];
u2(pi/4,3*pi/4) node[3];
cx node[1],node[3];
u1(-pi/4) node[3];
cx coin[0],node[3];
u1(pi/4) node[3];
cx node[1],node[3];
u1(-pi/4) node[3];
cx coin[0],node[3];
cu1(pi/8) coin[0],node[0];
cx coin[0],node[1];
cu1(-pi/8) node[1],node[0];
cx coin[0],node[1];
cu1(pi/8) node[1],node[0];
u2(pi/4,-pi) node[3];
cx node[2],node[3];
cx node[1],node[2];
cu1(-pi/8) node[2],node[0];
cx coin[0],node[2];
cu1(pi/8) node[2],node[0];
cx node[1],node[2];
u2(pi/8,-pi) node[1];
cu1(-pi/8) node[2],node[0];
cx coin[0],node[2];
p(pi/8) coin[0];
cu1(pi/8) node[2],node[0];
p(pi/8) node[2];
cx coin[0],node[2];
p(-pi/8) node[2];
cx coin[0],node[2];
u2(pi/8,3*pi/4) node[3];
cx node[2],node[3];
p(-pi/8) node[3];
cx coin[0],node[3];
p(pi/8) node[3];
cx node[2],node[3];
p(-pi/8) node[3];
cx coin[0],node[3];
cx node[3],node[1];
p(-pi/8) node[1];
cx node[2],node[1];
p(pi/8) node[1];
cx node[3],node[1];
p(-pi/8) node[1];
cx coin[0],node[1];
p(pi/8) node[1];
cx node[3],node[1];
p(-pi/8) node[1];
cx node[2],node[1];
p(pi/8) node[1];
cx node[3],node[1];
p(-pi/8) node[1];
cx coin[0],node[1];
ccx coin[0],node[3],node[2];
cx coin[0],node[3];
x coin[0];
u2(0,0) node[1];
x node[2];
x node[3];
cu1(pi/2) node[3],node[0];
u2(pi/4,-pi) node[3];
cx node[2],node[3];
u2(0,3*pi/4) node[3];
cx coin[0],node[3];
u1(pi/4) node[3];
cx node[1],node[3];
u1(-pi/4) node[3];
cx coin[0],node[3];
u1(pi/4) node[3];
cx node[1],node[3];
u2(pi/4,3*pi/4) node[3];
cx node[2],node[3];
u2(0,3*pi/4) node[3];
cu1(-pi/2) node[3],node[0];
u2(pi/4,-pi) node[3];
cx node[2],node[3];
u2(pi/4,3*pi/4) node[3];
cx node[1],node[3];
u1(-pi/4) node[3];
cx coin[0],node[3];
u1(pi/4) node[3];
cx node[1],node[3];
u1(-pi/4) node[3];
cx coin[0],node[3];
cu1(pi/8) coin[0],node[0];
cx coin[0],node[1];
cu1(-pi/8) node[1],node[0];
cx coin[0],node[1];
cu1(pi/8) node[1],node[0];
u2(pi/4,-pi) node[3];
cx node[2],node[3];
cx node[1],node[2];
cu1(-pi/8) node[2],node[0];
cx coin[0],node[2];
cu1(pi/8) node[2],node[0];
cx node[1],node[2];
u2(pi/8,-pi) node[1];
cu1(-pi/8) node[2],node[0];
cx coin[0],node[2];
p(pi/8) coin[0];
cu1(pi/8) node[2],node[0];
h node[0];
p(pi/8) node[2];
cx coin[0],node[2];
p(-pi/8) node[2];
cx coin[0],node[2];
u2(pi/8,3*pi/4) node[3];
cx node[2],node[3];
p(-pi/8) node[3];
cx coin[0],node[3];
p(pi/8) node[3];
cx node[2],node[3];
p(-pi/8) node[3];
cx coin[0],node[3];
cx node[3],node[1];
p(-pi/8) node[1];
cx node[2],node[1];
p(pi/8) node[1];
cx node[3],node[1];
p(-pi/8) node[1];
cx coin[0],node[1];
p(pi/8) node[1];
cx node[3],node[1];
p(-pi/8) node[1];
cx node[2],node[1];
p(pi/8) node[1];
cx node[3],node[1];
p(-pi/8) node[1];
cx coin[0],node[1];
ccx coin[0],node[3],node[2];
cx coin[0],node[3];
x coin[0];
u2(0,0) node[1];
x node[2];
x node[3];
barrier node[0],node[1],node[2],node[3],coin[0];
measure node[0] -> meas[0];
measure node[1] -> meas[1];
measure node[2] -> meas[2];
measure node[3] -> meas[3];
measure coin[0] -> meas[4];
