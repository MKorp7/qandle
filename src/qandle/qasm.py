from __future__ import annotations

import dataclasses
import re
import typing


def _ensure_tuple(value: typing.Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        value = value.strip()
        return (value,) if value else ()
    if isinstance(value, typing.Sequence):
        items: list[str] = []
        for element in value:
            items.extend(_ensure_tuple(element))
        return tuple(items)
    return (str(value),)


def _format_parameter(value: typing.Any) -> str:
    if isinstance(value, float):
        return f"{value:.16g}"
    return str(value)


@dataclasses.dataclass
class QasmRepresentation:
    """Structured representation of a circuit element for QASM emission."""

    gate_str: str
    gate_value: typing.Any = ""
    qubit: typing.Any = ""
    qasm3_inputs: typing.Any = ""
    qasm3_outputs: typing.Any = ""

    def __iter__(self):
        yield self

    def to_string(self) -> str:
        s = self.gate_str
        inputs = _ensure_tuple(self.qasm3_inputs)
        if inputs:
            s += "(" + ", ".join(inputs) + ")"
        else:
            if self.gate_value not in (None, ""):
                values = (
                    [self.gate_value]
                    if not isinstance(self.gate_value, typing.Sequence)
                    or isinstance(self.gate_value, str)
                    else list(self.gate_value)
                )
                rendered = ", ".join(_format_parameter(v) for v in values)
                s += f"({rendered})"
        if self.qubit not in (None, ""):
            if isinstance(self.qubit, typing.Sequence) and not isinstance(self.qubit, str):
                qubits = ", ".join(f"q[{q}]" for q in self.qubit)
                s += f" {qubits}"
            else:
                s += f" q[{self.qubit}]"
        s += ";"
        return s


class QASMNode(typing.Protocol):
    def to_qasm(self) -> str:
        ...


@dataclasses.dataclass(frozen=True)
class Program:
    version: str
    include_header: bool
    declarations: tuple[QASMNode, ...]
    io_declarations: tuple[QASMNode, ...]
    statements: tuple[QASMNode, ...]

    def to_qasm(self) -> str:
        lines: list[str] = []
        if self.include_header:
            lines.append(f"OPENQASM {self.version};")
            for decl in self.declarations:
                emitted = decl.to_qasm()
                if emitted:
                    lines.append(emitted)
        for io_decl in self.io_declarations:
            emitted = io_decl.to_qasm()
            if emitted:
                lines.append(emitted)
        for stmt in self.statements:
            emitted = stmt.to_qasm()
            if emitted:
                lines.append(emitted)
        return "\n".join(lines) + ("\n" if lines else "")


@dataclasses.dataclass(frozen=True)
class Declaration:
    keyword: str
    size: int
    name: str

    def to_qasm(self) -> str:
        return f"{self.keyword}[{self.size}] {self.name};"


@dataclasses.dataclass(frozen=True)
class IODeclaration:
    direction: str
    data_type: str
    name: str

    def to_qasm(self) -> str:
        return f"{self.direction} {self.data_type} {self.name};"


class Statement(typing.Protocol):
    def to_qasm(self) -> str:
        ...


@dataclasses.dataclass(frozen=True)
class Expression:
    value: str

    def to_qasm(self) -> str:
        return self.value


@dataclasses.dataclass(frozen=True)
class Operand:
    def to_qasm(self) -> str:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class RegisterOperand(Operand):
    name: str
    index: int

    def to_qasm(self) -> str:
        return f"{self.name}[{self.index}]"


@dataclasses.dataclass(frozen=True)
class RawOperand(Operand):
    value: str

    def to_qasm(self) -> str:
        return self.value


@dataclasses.dataclass(frozen=True)
class GateOperation:
    name: str
    arguments: tuple[Expression, ...]
    operands: tuple[Operand, ...]

    def to_qasm(self) -> str:
        argument_str = ""
        if self.arguments:
            argument_str = "(" + ", ".join(arg.to_qasm() for arg in self.arguments) + ")"
        operand_str = ""
        if self.operands:
            operand_str = " " + ", ".join(op.to_qasm() for op in self.operands)
        return f"{self.name}{argument_str}{operand_str};"


@dataclasses.dataclass(frozen=True)
class Measurement:
    qubit: Operand
    target: Operand

    def to_qasm(self) -> str:
        return f"measure {self.qubit.to_qasm()} -> {self.target.to_qasm()};"


@dataclasses.dataclass(frozen=True)
class Comment:
    text: str

    def to_qasm(self) -> str:
        return f"// {self.text}" if self.text else "//"


@dataclasses.dataclass(frozen=True)
class RawStatement:
    text: str

    def to_qasm(self) -> str:
        return self.text


_MEASURE_RE = re.compile(
    r"^measure\s+(?P<qubit>[A-Za-z_][\w]*(?:\[\d+\]))\s*->\s*(?P<bit>[A-Za-z_][\w]*(?:\[\d+\]))$"
)
_GATE_RE = re.compile(r"^(?P<name>[A-Za-z_][\w]*)(?P<rest>.*)$")
_INDEXED_RE = re.compile(r"^(?P<name>[A-Za-z_][\w]*)\[(?P<index>\d+)\]$")


def _parse_operand(text: str) -> Operand:
    match = _INDEXED_RE.fullmatch(text)
    if match:
        return RegisterOperand(match.group("name"), int(match.group("index")))
    return RawOperand(text)


def _normalize_operands(value: typing.Any) -> tuple[Operand, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, typing.Sequence) and not isinstance(value, str):
        ops: list[Operand] = []
        for element in value:
            ops.extend(_normalize_operands(element))
        return tuple(ops)
    if isinstance(value, int):
        return (RegisterOperand("q", value),)
    if isinstance(value, str):
        return (_parse_operand(value),)
    return (RawOperand(str(value)),)


def _extract_inline_arguments(rest: str) -> tuple[tuple[Expression, ...], str]:
    stripped = rest.lstrip()
    if not stripped.startswith("("):
        return (), rest.strip()
    depth = 0
    for idx, ch in enumerate(stripped):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                inside = stripped[1:idx]
                remainder = stripped[idx + 1 :].strip()
                args = tuple(
                    Expression(token.strip())
                    for token in inside.split(",")
                    if token.strip()
                )
                return args, remainder
    return (), rest.strip()


def _collect_parameters(rep: QasmRepresentation) -> tuple[Expression, ...]:
    inputs = _ensure_tuple(rep.qasm3_inputs)
    if inputs:
        return tuple(Expression(name) for name in inputs)
    value = rep.gate_value
    if value in (None, ""):
        return ()
    if isinstance(value, typing.Sequence) and not isinstance(value, str):
        return tuple(Expression(_format_parameter(v)) for v in value)
    return (Expression(_format_parameter(value)),)


def _representation_to_statement(rep: QasmRepresentation) -> Statement:
    text = rep.gate_str.strip()
    if not text:
        return RawStatement("")
    if text.startswith("//"):
        return Comment(text[2:].strip())
    normalized = text.rstrip(";")
    measure_match = _MEASURE_RE.fullmatch(normalized)
    if measure_match:
        qubit = _parse_operand(measure_match.group("qubit"))
        target = _parse_operand(measure_match.group("bit"))
        return Measurement(qubit, target)
    if normalized.startswith("gate "):
        return RawStatement(text)
    gate_match = _GATE_RE.match(normalized)
    if not gate_match:
        return RawStatement(text)
    name = gate_match.group("name")
    rest = gate_match.group("rest")
    inline_args, remaining = _extract_inline_arguments(rest)
    operands: tuple[Operand, ...]
    if remaining:
        operands = tuple(
            _parse_operand(token.strip())
            for token in remaining.split(",")
            if token.strip()
        )
    else:
        operands = _normalize_operands(rep.qubit)
    params = inline_args if inline_args else _collect_parameters(rep)
    if not inline_args:
        params = _collect_parameters(rep)
    else:
        params = inline_args
        extra = _collect_parameters(rep)
        if extra:
            params = params + extra
    return GateOperation(name=name, arguments=params, operands=operands)


def _build_openqasm3_program(
    num_qubits: int,
    reps: list[QasmRepresentation],
    include_header: bool,
) -> Program:
    statements = tuple(_representation_to_statement(rep) for rep in reps)
    input_names: set[str] = set()
    output_names: set[str] = set()
    for rep in reps:
        input_names.update(_ensure_tuple(rep.qasm3_inputs))
        output_names.update(_ensure_tuple(rep.qasm3_outputs))
    declarations: tuple[QASMNode, ...] = (
        Declaration("qubit", num_qubits, "q"),
        Declaration("bit", num_qubits, "c"),
    )
    io_declarations: list[IODeclaration] = []
    for name in sorted(input_names):
        io_declarations.append(IODeclaration("input", "float", name))
    for name in sorted(output_names):
        io_declarations.append(IODeclaration("output", "float", name))
    return Program(
        version="3.0",
        include_header=include_header,
        declarations=declarations,
        io_declarations=tuple(io_declarations),
        statements=statements,
    )


def _emit_openqasm2(
    num_qubits: int, reps: list[QasmRepresentation], include_header: bool
) -> str:
    header = """OPENQASM 2.0;
include \"qelib1.inc\";
qreg q[{num_qubits}];
creg c[{num_qubits}];
"""
    body = "\n".join(rep.to_string() for rep in reps)
    if include_header:
        return header.format(num_qubits=num_qubits) + body + ("\n" if body else "")
    return body + ("\n" if body else "")


def _emit_openqasm3(
    num_qubits: int, reps: list[QasmRepresentation], include_header: bool
) -> str:
    program = _build_openqasm3_program(num_qubits, reps, include_header)
    if include_header:

        outp += qasm_header
    if qasm_version == 3:
        declared_inputs = set()
        declared_outputs = set()
        input_lines = []
        output_lines = []

        for rep in reps:
            for r in rep:
                if r.qasm3_inputs != "":
                    for name in r.qasm3_inputs.split(","):
                        name = name.strip()
                        if name and name not in declared_inputs:
                            input_lines.append(f"input float {name};\n")
                            declared_inputs.add(name)
                if r.qasm3_outputs != "":
                    for name in r.qasm3_outputs.split(","):
                        name = name.strip()
                        if name and name not in declared_outputs:
                            output_lines.append(f"output float {name};\n")
                            declared_outputs.add(name)

        outp += "".join(input_lines + output_lines)


def convert_to_qasm(circuit, qasm_version: int = 2, include_header: bool = True):
    """Convert a circuit to OpenQASM 2 or 3."""

    num_qubits = circuit.num_qubits
    reps_nested = circuit.to_qasm()
    reps: list[QasmRepresentation] = []
    for rep in reps_nested:
        reps.extend(rep)
    if qasm_version == 2:
        return _emit_openqasm2(num_qubits, reps, include_header)
    if qasm_version == 3:
        return _emit_openqasm3(num_qubits, reps, include_header)
    raise ValueError(f"Unsupported OpenQASM version: {qasm_version}")
