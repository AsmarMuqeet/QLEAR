// Benchmark was created by MQT Bench on 2023-06-29
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: v1.0.0
// Qiskit version: {'qiskit-terra': '0.24.1', 'qiskit-aer': '0.12.0', 'qiskit-ignis': None, 'qiskit-ibmq-provider': '0.20.2', 'qiskit': '0.43.1', 'qiskit-nature': '0.6.2', 'qiskit-finance': '0.3.4', 'qiskit-optimization': '0.5.0', 'qiskit-machine-learning': '0.6.1'}

OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg meas[2];
ry(2.14242922916416) q[0];
ry(-1.94440140694857) q[1];
cx q[0],q[1];
ry(1.20774261740146) q[0];
ry(-1.22175081018962) q[1];
cx q[0],q[1];
ry(0.75527772195187) q[0];
ry(-2.25181117701188) q[1];
cx q[0],q[1];
ry(1.56365716628813) q[0];
ry(2.22238748160754) q[1];
barrier q[0],q[1];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
