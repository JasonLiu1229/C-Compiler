grammar File;

// Grammar for subset of C language
file: (comment* incl_stat)* (
		comment
		| instr
		| func_defn ((';')* | DELIM)
		| func_decl ((';')+ | DELIM)
	)* EOF;

// Include statement
incl_stat: INCLUDE '<' library = VAR_NAME '.h' '>';

// Global instructions
instr:
	declr ((';')+ | DELIM) // Global declaration
	| array_decl ((';')+ | DELIM) // Global array declaration
	| expr ((';')+ | DELIM) // Global expression
	| assign ((';')+ | DELIM) // Global assignment
	| scope // Global scope
	| enum_declr ((';')+ | DELIM); // Global enum declaration

//  Variable declaration
declr: CONST? TYPE (var_decl ',')* var_decl;

// Printf instruction
printf:
	PRINTF '(' (rvar | rtype | print_val = STRING) ')'
	| PRINTF '(' (
		format_string = SCANF_STRING
		| format_string = STRING
	) (',' (vars += printf_arg ',')* vars += printf_arg)? ')';

// Printf arguments. Can be a variable, dereference, array element, expression or a inputStr literal
printf_arg:
	rvar
	| rtype
	| array_el
	| deref
	| comp
	| expr
	| STRING;

// Scanf instruction. Format inputStr followed by variables to be scanned
scanf:
	SCANF '(' (
		format_string = SCANF_STRING
		| format_string = STRING
	) ',' (ADDR? (vars += scanf_arg) ',')* ADDR? vars += scanf_arg ')';

scanf_arg: rvar | deref | array_el | ADDR rvar | ADDR deref;

// Comments
comment: com = COMMENT | com = LCOMMENT;

enum_declr:
	ENUM VAR_NAME '{' (enum_val += VAR_NAME ',')* enum_val += VAR_NAME '}';

// TODO: #define statement : #define old_value new_value

// TODO: typedef statement : typedef old_type new_type

// TODO: structs : struct_name { struct_members }. Members can be any type, optional default value

// TODO: unions : union_name { union_members }. Members can be any type, optional default value

// TODO: function pointers : return_type (*function_name)(param_type1, param_type2, ...)

// TODO: fgets, fput, fopen, fclose

// Function definition, declaration and call
func_defn:
	const = CONST? (type = TYPE | type = VOID) ptr += STR* name = VAR_NAME '(' params = param_list?
		')' func_scope;

func_decl:
	const = CONST? (type = TYPE | type = VOID) ptr += STR* name = VAR_NAME '(' params = param_list?
		')';

param_list: params += param_declr (',' params += param_declr)*;

param_declr:
	const = CONST? type = TYPE (
		reference = ADDR? ptr += STR*
		| ptr += STR* reference = ADDR?
	) var = VAR_NAME (ASSIGN default = expr)?;

func_call: name = VAR_NAME '(' args = arg_list? ')';

arg_list: args += func_arg (',' args += func_arg)*?;

func_arg: rvar | deref | func_call | rtype | expr;

// Function scope
func_scope:
	'{' (
		printf ((';')+ | DELIM)
		| scanf ((';')+ | DELIM)
		| if_cond ((';')* | DELIM)
		| switch_instr ((';')* | DELIM)
		| while_loop ((';')* | DELIM)
		| for_loop ((';')* | DELIM)
		| assign ((';')+ | DELIM)
		| comp ((';')+ | DELIM)
		| instr
		| return_instr
		| comment
	)* '}';

// Scope (in case of if, else, while, for, switch) and global scope
scope:
	'{' (
		printf ((';')+ | DELIM)
		| scanf ((';')+ | DELIM)
		| if_cond ((';')* | DELIM)
		| switch_instr ((';')* | DELIM)
		| while_loop ((';')* | DELIM)
		| for_loop ((';')* | DELIM)
		| assign ((';')+ | DELIM)
		| comp ((';')+ | DELIM)
		| break_instr
		| cont_instr
		| instr
		| return_instr
		| comment
	)* '}';

return_instr:
	RETURN (ret_val = expr)? ';' (instr | return_instr)*;

// Switch
switch_instr:
	SWITCH '(' switch_cond = expr ')' '{' (
		case_list += case_instr
	)* default = default_instr? '}';

case_instr: CASE case_cond = expr ':' switch_scope;

default_instr: DEFAULT ':' switch_scope;

switch_scope: (
		printf ((';')+ | DELIM)
		| scanf ((';')+ | DELIM)
		| if_cond ((';')* | DELIM)
		| switch_instr ((';')* | DELIM)
		| while_loop ((';')* | DELIM)
		| for_loop ((';')* | DELIM)
		| assign ((';')+ | DELIM)
		| comp ((';')+ | DELIM)
		| break_instr
		| cont_instr
		| instr
		| return_instr
		| comment
	)*;

cont_instr: CONTINUE (';' | DELIM) instr*;

break_instr: BREAK (';' | DELIM) instr*;

// Control structures: if, else, while, for
if_cond: IF '(' condition = cond ')' scope else_cond?;

else_cond: ELSE scope;

while_loop: WHILE '(' condition = cond ')' scope;

for_loop:
	FOR '(' initialization = init ';' condition = cond ';' increment = incr ')' scope;

// Variable initialization (declaration and assignment)
init: TYPE lvar ASSIGN expr | assign;

// Array declaration
array_decl:
	const = CONST? type = TYPE ptr += STR* name = VAR_NAME '[' size = INT? ']' ASSIGN '{' (
		values += rtype ','
	)* values += rtype '}' // Array declaration with initialization
	| const = CONST? type = TYPE ptr += STR* name = VAR_NAME '[' size = INT ']';
// Array declaration without initialization (size must be specified)

// comparison takes precedence over mathemathical operations
cond: comp | expr;

incr:
	INCR rvar // Increment
	| DECR rvar // Decrement
	| rvar INCR // Increment (postfix)
	| rvar DECR; // Decrement (postfix)

// Right-hand side variable use
var_decl: lvar (ASSIGN expr)?;

assign: (rvar | deref | array_el) ASSIGN expr;

array_el: (lvar | deref) '[' (index = INT | expr) ']';

deref: STR (deref | rvar); // Dereference can be chained

lvar: ptr += STR* name = VAR_NAME;

rvar: name = VAR_NAME;

comp:
	lhs = expr op = (
		AND_OP
		| OR_OP
		| GEQ
		| LEQ
		| NEQ
		| GT
		| LT
		| EQ
	) rhs = expr;

expr:
	rhs = term
	| oper1 = expr op = (SUM | DIF | AND_OP | OR_OP) oper2 = term;

// 'term' rule: Differentiates unary and binary operations. Binary: Uses 'left_term' and
// 'right_factor'. Unary: Uses 'single_factor' or 'single_term'.
term:
	left_term = term op = (
		STR
		| DIV
		| MOD
		| GT
		| LT
		| GEQ
		| LEQ
		| EQ
		| NEQ
	) right_factor = factor
	| single_factor = factor
	| op = NOT_OP single_factor = factor
	| single_term = term op = (INCR | DECR);
// declare precedence of operations
factor:
	single_primary = primary
	| op = (SUM | DIF | INCR | DECR) single_factor = factor;

// Primary expressions (highest precedence)
primary:
	rvar
	| rtype
	| addr = ADDR var = rvar
	| deref
	| '(' in_expr = expr ')'
	| cast_expr = CAST cast_val = primary
	| func_call
	| array_el;

// type of right-hand side variable
rtype:
	type = INT
	| type = FLOAT
	| type = CHAR
	| type = BOOL
	| STRING;
// TODO: Support for strings

// Keywords
CAST: '(' TYPE ')';
CONST: 'const';
IF: 'if';
ELSE: 'else';
FOR: 'for';
WHILE: 'while';
BREAK: 'break';
CONTINUE: 'continue';
SWITCH: 'switch';
CASE: 'case';
DEFAULT: 'default';
PRINTF: 'printf';
RETURN: 'return';
SCANF: 'scanf';
INCLUDE: '#include';
// Identifiers and data types
TYPE: 'char' | 'float' | 'int' | 'bool';
// Types
VOID: 'void';
ENUM: 'enum';
VAR_NAME: ([a-z] | [A-Z] | '_') ([a-z] | [A-Z] | [0-9] | '_')*; // match lower-case identifiers
INT: ([1-9][0-9]*) | [0];
FLOAT: [0-9]+ '.' [0-9]+;
CHAR: ('\'' . '\'') | ('\'\\' . '\'');
BOOL: 'true' | 'false';
SCANF_STRING: '"' ('%' ('-' | '+')? (INT)? [discf])* '"';
STRING: '"' (.)*? '"';

// Operations
STR: '*';
DIV: '/';
MOD: '%';
SUM: '+';
DIF: '-';
LT: '<';
LEQ: '<=';
GT: '>';
GEQ: '>=';
EQ: '==';
NEQ: '!=';
OR_OP: '||';
AND_OP: '&&';
NOT_OP: '!';
ASSIGN: '=';
ADDR: '&';
INCR: '++';
DECR: '--';
// Redundant characters to be removed
SP: [ ]+ -> skip;
NEWLINE: [\r\n]+ -> skip;
WS: [ \t\r\n]+ -> skip; // skip spaces, tabs, newlines
UNICODE_WS:
	[\p{White_Space}] -> skip; // match all Unicode whitespace
LN: [ \t\n]+ -> skip; // skip spaces, tabs, newlines
DELIM: [;]+; // Delimiter for statements. One or more semicolons
// Comments Ref : https://stackoverflow.com/a/23414078
COMMENT: '/*' .*? '*/';
LCOMMENT: '//' ~[\r\n]*;