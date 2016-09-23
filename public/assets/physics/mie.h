/*
	mie.h
	Author: Jesse Laning - Kansas State University
	Email: jlaning@k-state.edu

	############################################################

	This program iterates along a lattice through a sphere
	calculating the electric field part of the electromagnetic
	field using the known Mie solutions to Maxwell's equations
	from a plane wave on the z-axis.

	Using this data it can generate intensity vs angle/qR plots
	or it can add up shells of the field normalized by the
	volume of the shell.

	Runtime arguements:
	-o      Sets the output file prefix for the final data. Default is nothing. 					Usage -o prefix
	-p      Sets the number of parallel threads to use. Default is the number of hardware threads. 	Usage -p threads (a positive integer)
	-r      Sets the radius of the sphere. Default is 5. 											Usage -r radius (a positive real number)
	-s      Sets the step size through the sphere. Default is 0.1. 									Usage -s stepsize (a positive real number)
	-t      Sets the number of angle steps through the sphere. Default is 1000. 					Usage -t steps (a positive integer)
	-l      Sets the wavelength of the incident field. Default is pi/6. 							Usage -l wavelength (a positive real number)
	-m      Sets the real part of the index of refraction. Default is 1.5. 							Usage -m threads (a positive real number greater or equal to 1)
	-k      Sets the imaginary part of the index of refraction. Default is 0.0. 					Usage -r threads (a positive real number)
	-q      Adds a q value, can be used like a list. 												Usage "-q 0 1 2 3 4 5 6"
	-qr     Adds a qR value, can be used like a list. 												Usage "-qr 0 1 2 3 4 5 6"
	--intensity-vs-angle    Flag for calculating intensity vs angle
	--intensity-vs-qr       Flag for calculating intensity vs qR
	--shell					Flag for calculating Spherical Shells
	--half-shell			Flag to use half shells, these are like shells but they are split into Hemispherical Shells where z = 0
	--center				Flag to calculate the innermost shell only
	--edge					Flag to calculate the outermost shell only
	--sum-shell				Flag to sum all of the shells together
	--field					Flag to output just the field, this is just if it is running on the z-axis or using --shell or --half-shell
	--gnuplot				Flag for generating gnuplot files
	--gnuonly				Flag for generating only gnuplot files
	--nonorm				Flag to not normalize data to 1 (helpful for generating multiple plots that you want to compare by magnitude)
	(--)help/h/?			Shows help page (the above info)

	This program can also be compiled to only probe along the
	z-axis.	This lets you look at the electric field along just
	the z-axis. To do this, compile for the z-axis only and then
	use the list of runtime arguments that deal with shells but
	instead of shells the data is just points along the z-axis
	where half shells are the field at their point along the
	z-axis while shells are the sum of the fields at their
	radius from the center of the spehere.

	############################################################

	Can compile with a minimum of g++ 4.9.2 or Microsoft Visual C++ 2015

	To compile on linux use:

			g++ -Wall --std=gnu++11 -I./ -lpthread mie.cpp -o mie

	or to compile on linux to probe just the z-axis use:

			g++ -Wall --std=gnu++11 -I./ -lpthread mie.cpp -o mie_z -DZ_AXIS_ONLY
*/

#pragma once
#include <algorithm>
#include <chrono>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <string.h>
#include <thread>
#include <vector>

#ifndef EXIT_SUCCESS
#define EXIT_SUCCESS 0
#endif
#ifndef EXIT_FAILURE
#define EXIT_FAILURE 1
#endif

#ifndef COMMA_SEPARATED
#define COMMA_SEPARATED string(".csv")
#endif
#ifndef PLOT
#define PLOT string(".plt")
#endif
#ifndef VECTORS
#define VECTORS string("-vectors")
#endif
#ifndef LINES
#define LINES string("-lines")
#endif
#ifndef DATA_HEADER
#define DATA_HEADER "#   R\t\tRealm\tKappa\t\tk\t\tqR\t\tr\t\tRealE\t\tImagE"
#endif

using namespace std;

#define TO_DBL(a) static_cast<double>(a)
#define TO_INT(a) static_cast<int>(a)
#define TO_LDBL(a) static_cast<long double>(a)
#define TO_LNG(a) static_cast<long>(a)
#define TO_UINT(a) static_cast<unsigned>(a)

typedef std::complex<double> Complex;
typedef Complex IndexOfRefraction;

#ifdef Z_AXIS_ONLY
const string gnu_title = "Z-axis";
const string half_gnu_title = "Z-axis";
#else
const string gnu_title = "Full Shell";
const string half_gnu_title = "Half Shell";
#endif

//3.141592653589793238462643383279502884L
constexpr long double pi = 3.141592653589793238462643383279502884L;
//2.0L * 3.141592653589793238462643383279502884L
constexpr long double pi2 = pi * 2.0L;
//sqrt(-1)
constexpr Complex imaginary_unit(0.0, 1.0);
//seperator for data in file output
constexpr char outputDelimiter = ',';
//Double point numerical limit
constexpr numeric_limits<double> dblLimit;

enum DataType
{
	INTENSITY_VS_ANGLE,
	INTENSITY_VS_QR,
	FULL_SHELL,
	HALF_SHELL
};

enum SUM_TYPE
{
	POSITIVE,
	NEGATIVE,
	BOTH
} _sumType;

//Runtime options that can be set through console arguements
struct Options
{
public:
	//Flag to output just the field, this is just if it is running on the z-axis
	bool _field{ false };
	//Flag to calculate the outermost shell only
	bool _edge{ false };
	//Flag to sum all of the shells together
	bool _sumShell{ false };
	//Flag to normalize the data to 1
	bool _normalize{ true };
	//Flag to calculate the innermost shell only
	bool _center{ false };
	//Flag to output half shells instead of full shells. The split is the xy-plane at the origin
	bool _halfShell{ false };
	//Flag to calculate shells
	bool _shell{ false };
	//Flag to only generate gnuplot files
	bool _gnuonly{ false };
	//Flag to generate gnuplot .plt files for easy graphing
	bool _gnuplot{ false };
	//Flag to calculate perpendicular intensity vs scattered angle
	bool _iVA{ false };
	//Flag to calculate intensity vs qR
	bool _iVqR{ false };
	//Radius of sphere
	double _r{ 5.0 };
	//Step size through sphere
	double _stepsize{ 0.1 };
	//Step size through scattered angle
	int _thetastep{ 1000 };
	//Wave length
	double _lambda{ TO_DBL(pi / 6.0L) };
	//particle size parameter
	double _k{ TO_DBL(pi2 / TO_LDBL(_lambda)) };
	double _x{ _k * _r };
	//Real part of index of refraction
	double _realm{ 1.5 };
	//Imaginary part of index of refraction
	double _kappa{ 0.0 };
	//Index of refraction
	IndexOfRefraction _m{ _realm, _kappa };
	Complex _mx{ _m * _x };

	//Output file prefix
	string _file{ "" };

	//Number of parallel threads
	int _parallelthreads = thread::hardware_concurrency();

	//These control the recursion formulas for calculation of special functions and should not be changed
	int _xstop{ TO_INT(_x + 4.E0 * pow(_x, 0.3333) + 20.0) };
	double _ymod{ abs(_mx) };
	int _nmax{ TO_INT(max(TO_DBL(_xstop), _ymod) + 15.0) };

	void update()
	{
		_k = TO_DBL(pi2 / TO_LDBL(_lambda));
		_x = _k * _r;
		_m = { _realm, _kappa };
		_mx = { _m * _x };
		_xstop = TO_INT(_x + 4.E0 * pow(_x, 0.3333) + 20.0);
		_ymod = abs(_mx);
		_nmax = TO_INT(max(TO_DBL(_xstop), _ymod) + 15.0);
	
	#ifdef Z_AXIS_ONLY
		_parallelthreads = 1;
		if(_file.empty())
		{
			_file.append("z-axis");
		}
		else
		{
			_file = "z-axis" + _file;
		}
		if(_field) _halfShell = true;
	#endif

		if (!_file.empty()) _file.append("-");

		_file.append("R_").append(to_string(_r));
		_file.append("-S_").append(to_string(_stepsize));
		_file.append("-L_").append(to_string(_lambda));
		_file.append("-M_").append(to_string(_realm));
		_file.append("-K_").append(to_string(_kappa));
		
		if(_center && (_shell || _halfShell))
		{
			_r = 2.0 * _stepsize;
			_parallelthreads = 1;
		}
		
		if(_edge && (_shell || _halfShell))
		{
			_parallelthreads = 2;
		}
	}
} _options;

template<typename T> struct Point
{
	T _x, _y, _z;

	bool operator<(const Point<T>& p) const
	{
		if (_x != p._x) return _x < p._x;
		else if (_y != p._y) return _y < p._y;
		return _z < p._z;
	}

	Point<T> operator-(const Point<T>& p) const
	{
		return { _x - p._x, _y - p._y, _z - p._z };
	}

	Point<T> operator+(const Point<T>& p) const
	{
		return{ _x + p._x, _y + p._y, _z + p._z };
	}

	template<typename t> Point<t> Convert(t delimiter)
	{
		Point<t> r;
		r._x = static_cast<t>(_x) * delimiter;
		r._y = static_cast<t>(_y) * delimiter;
		r._z = static_cast<t>(_z) * delimiter;
		return r;
	}

	double abs()
	{
		return sqrt(TO_DBL(_x * _x + _y * _y + _z * _z));
	}
};

template<typename T> struct SphereShell
{
	T _start;
	T _end;
	
	Complex _negativeSum;
	Complex _positiveSum;
	
	long _nNegativePoints{ 0 };
	long _nPositivePoints{ 0 };

	double abs()
	{
		return std::abs(totalSum());
	}

	int intRadius()
	{
		return TO_INT((TO_DBL(_end - _start) / 2.0) + TO_DBL(_start));
	}

	double radius()
	{
		return TO_DBL(intRadius()) * _options._stepsize;
	}

	Complex totalSum()
	{
		switch (_sumType)
		{
		case SUM_TYPE::POSITIVE:
			return _positiveSum;
		case SUM_TYPE::NEGATIVE:
			return _negativeSum;
		case SUM_TYPE::BOTH:
			return _positiveSum + _negativeSum;
		}
		return Complex(0, 0);
	}
};

template<class _Ty> double abs(SphereShell<_Ty>& s) { return s.abs(); }

struct ArgumentHandler
{
	string _option;
	string _help;
	string _usage;

	enum ArgumentType
	{
		STRING,
		BOOL,
		INT,
		DOUBLE,
		CUSTOM
	} _type;

	union
	{
		int* _i;
		double* _d;
		bool* _b;
		string* _str;
	} _data;

	function<bool()> _valid;

	ArgumentHandler(string option, string help, string usage, ArgumentType t, int* i, bool valid() = [] { return true; }) : ArgumentHandler(option, help, usage, t, valid) { _data._i = i; }
	ArgumentHandler(string option, string help, string usage, ArgumentType t, double* d, bool valid() = [] { return true; }) : ArgumentHandler(option, help, usage, t, valid) { _data._d = d; }
	ArgumentHandler(string option, string help, string usage, ArgumentType t, bool* b, bool valid() = [] { return true; }) : ArgumentHandler(option, help, usage, t, valid) { _data._b = b; }
	ArgumentHandler(string option, string help, string usage, ArgumentType t, string* str, bool valid() = [] { return true; }) : ArgumentHandler(option, help, usage, t, valid) { _data._str = str; }

private:
	ArgumentHandler(string option, string help, string usage, ArgumentType t, bool valid() = [] { return true; }) : _option(option), _help(help), _usage(usage), _type(t), _valid(valid) {}
};

vector<Complex> _partialSums;
#ifdef Z_AXIS_ONLY
vector<pair<double, vector<Complex>>> _zAxisShells;
#else
//The best kind of pasta
vector<pair<double, vector<SphereShell<long>>>> _shells;
mutex _fieldLock;
#endif

static decltype(chrono::high_resolution_clock::now()) now() { return chrono::high_resolution_clock::now(); }

int handleArguments(int argc, const char* argv[]);

void describeRun();
void writeDataToFile();
void writeGnuplotFiles();

void calcAnx(vector<Complex>& Anx, int nmax, double x);
void calcAnmx(vector<Complex>& Anmx, int nmax, const Complex& mx);
void calcBnx(vector<Complex>& Bnx, int nmax, double x);
void calcBnmx(vector<Complex>& Bnmx, int nmax, const Complex& mx);
void calcPsinx(vector<Complex>& Anx, vector<Complex>& psinx, int nmax, double x);
void calcPsinmx(vector<Complex>& Anmx, vector<Complex>& psinmx, int nmax, const Complex& mx);
void calcXix(vector<Complex>& Bnx, vector<Complex>& xix, int nmax, double x);
void calcXinmx(vector<Complex>& Bnmx, vector<Complex>& xinmx, int nmax, const Complex& mx);

Complex En(int n);
Complex Enr(int n);

void calcCartesianFields(int start, int length, int steps);

void tryMakeDir(string dir, int o = 0700, bool verbose = false);
string filename(double q, DataType type, string extension, bool inclDir = true);
string leadingZeros(double);

static ofstream& operator<<(std::ofstream& o, const ofstream& s) { return o; }

template<typename T> static double normalizer(vector<T>& p)
{
	if (_options._center || !_options._normalize) return 1.0;
	return abs(*max_element(p.begin(), p.end(),
	[](T& a, T& b)
	{
		return abs(a) < abs(b);
	}));
}

static ofstream& writeDataInfo(ofstream& out)
{
	out << to_string(_options._r)
		<< outputDelimiter << to_string(_options._m.real())
		<< outputDelimiter << to_string(_options._m.imag())
		<< outputDelimiter << to_string(_options._k);
	return out;
}

template<typename T> static ofstream& writeData(ofstream& out, int i, Complex s, pair<double, vector<T>>& p)
{
	auto pre = out.precision();
	double smax = normalizer(p.second);
	out << writeDataInfo(out)
		<< outputDelimiter << to_string(p.first * _options._r)
		<< outputDelimiter << to_string(TO_DBL(i) * _options._stepsize)
		<< outputDelimiter << setprecision(dblLimit.max_digits10) << s.real() / smax
		<< outputDelimiter << s.imag() / smax << setprecision(pre);
	return out;
}
