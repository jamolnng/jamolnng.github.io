/*
	mie.cpp
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
	-qr     Adds a qR value, can be used like a list. 												Usage "-q 0 1 2 3 4 5 6"
	--intensity-vs-angle    Flag for calculating intensity vs angle
	--intensity-vs-qr       Flag for calculating intensity vs qR
	--shell					Flag for calculating Spherical Shells
	--half-shell			Flag to use half shells, these are like shells but they are split into Hemispherical Shells where z = 0
	--center				Flag to calculate the innermost shell only
	--edge					Flag to calculate the outermost shell only
	--sum-shell				Flag to sum all of the shells together, this is just if it is running on the z-axis
	--field					Flag to output just the field
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

#include "mie.h"

#ifdef Z_AXIS_ONLY
#ifdef _WIN32
#pragma message("Compiling to run the code for the z-axis only!")
#else
#warning "Compiling to run the code for the z-axis only!"
#endif
#endif

int main(int argc, const char* argv[])
{
	//Handle runtime arguements, if there was an error, exit the program
	if (handleArguments(argc, argv)) exit(EXIT_FAILURE);
	
	//Get the time the program started running.
	auto start = now();
	
	//Make output directories based on runtime arguments
	if(_options._iVA || _options._iVqR) tryMakeDir("./out/" + to_string(_options._m.real()));
	
#ifdef Z_AXIS_ONLY
	if (_options._halfShell) tryMakeDir("./out/Z-axis/Half Shell/" + to_string(_options._m.real()));
	if (_options._shell) tryMakeDir("./out/Z-axis/Full Shell/" + to_string(_options._m.real()));
#else
	if (_options._halfShell) tryMakeDir("./out/Half Shell/" + to_string(_options._m.real()));
	if (_options._shell) tryMakeDir("./out/Full Shell/" + to_string(_options._m.real()));
#endif

	//write basic experiement info to console
	describeRun();
	
	//Container for all concurrent threads
	vector<thread> threads(_options._parallelthreads - 1);

	if (_options._iVA || _options._iVqR) _partialSums = vector<Complex>(_options._thetastep + 1);

	//Variables to determine the portion of the sphere that each thread will calculate
	unsigned steps = TO_UINT(_options._r / _options._stepsize);
	unsigned m = steps % _options._parallelthreads;
	unsigned offset = 0;
	unsigned length = (steps - m) / _options._parallelthreads;
	unsigned l = 0;
	
	//Skip to writing gnuplot files if flag --gnuonly is used
	if(_options._gnuonly) goto gnuplot;
	
	//Start each concurrent thread
	if(_options._edge)
	{
		threads.push_back(thread(calcCartesianFields, 0, 3, 2));
		calcCartesianFields(steps - 2, 3, 2);
	}
	else
	{
		for (auto i = 0; i < _options._parallelthreads; i++)
		{
			l = length;
			if (m > 0)
			{
				l++;
				m--;
			}
			if (i == _options._parallelthreads - 1)
			{
				l++;
				calcCartesianFields(offset, l, steps);
			}
			else
			{
				threads.push_back(thread(calcCartesianFields, offset, l, steps));
			}
			offset += l;
		}
	}
	
	//Wait for all concurrent threads to finish then free them in memory
	if(_options._parallelthreads > 1)
	{
		for (auto& t : threads) if (t.joinable()) t.join();
		threads.clear();
		threads.shrink_to_fit();
	}
	
	//Write generated data to file
	writeDataToFile();
	
	gnuplot:
	//Write gnuplot files
	if (_options._gnuplot || _options._gnuonly) writeGnuplotFiles();
	
	//Get the program end time and write to the console how long the program took to run in seconds
	auto end = now();
	cout << "Done in " << chrono::duration<double>(end - start).count() << " seconds." << endl;

	return EXIT_SUCCESS;
}

int handleArguments(int argc, const char* argv[])
{
	vector<ArgumentHandler> argumentHandlers;
	auto add = [&](ArgumentHandler a) { argumentHandlers.push_back(a); };
	add({ "-o", "Sets the output file prefix for the final data. Default is nothing", "output prefix. Usage is: '-o prefix'", ArgumentHandler::STRING, &_options._file });
	add({ "-p", "Sets the number of parallel threads to use. Default is the number of hardware threads", "number of threads. Usage is: '-p threads', threads must be greater than zero", ArgumentHandler::INT, &_options._parallelthreads, [] { return _options._parallelthreads > 0; } });
	add({ "-r", "Sets the radius of the sphere. Default is 5", "radius. Usage is: '-r radius', radius must be greater than zero", ArgumentHandler::DOUBLE, &_options._r, [] { return _options._r > 0; } });
	add({ "-s", "Sets the step size through the sphere. Default is 0.1", "step size. Usage is: -s 'stepsize', stepsize must be greater than zero", ArgumentHandler::DOUBLE, &_options._stepsize, [] { return _options._stepsize > 0; } });
	add({ "-t", "Sets the number of angle steps through the sphere. Default is 1000", "number of theta steps. Usage is: -t 'steps', steps must be greater than zero", ArgumentHandler::INT, &_options._thetastep, [] { return _options._thetastep > 0; } });
	add({ "-l", "Sets the wavelength of the incident field. Default is pi/6", "wavelength. Usage is: '-l wavelength', wavelength must be greater than zero", ArgumentHandler::DOUBLE, &_options._lambda, [] { return _options._lambda > 0; } });
	add({ "-m", "Sets the real part of the index of refraction. Default is 1.5", "real part of index of refraction. Usage is: '-m real', real must be greater than zero", ArgumentHandler::DOUBLE, &_options._realm, [] { return _options._realm > 0; } });
	add({ "-k", "Sets the imaginary part of the index of refraction. Default is 0.0", "imaginary part of index of refraction. Usage is: '-k imaginary', imaginary must be greater than zero", ArgumentHandler::DOUBLE, &_options._kappa, [] { return _options._kappa >= 0; } });
	add({ "-q", "Adds a q value", "q value. Usage is: '-q values'", ArgumentHandler::CUSTOM, (double*)nullptr });
	add({ "-qr", "Adds a qR value", "qR value. Usage is: '-qR values'", ArgumentHandler::CUSTOM, (double*)nullptr });
	add({ "--intensity-vs-angle", "Flag for calculating intensity vs angle", "", ArgumentHandler::BOOL, &_options._iVA });
	add({ "--intensity-vs-qr", "Flag for calculating intensity vs qR", "", ArgumentHandler::BOOL, &_options._iVqR });
	add({ "--shell", "Flag for calculating shells", "", ArgumentHandler::BOOL, &_options._shell });
	add({ "--half-shell", "Flag to use half shells", "", ArgumentHandler::BOOL, &_options._halfShell });
	add({ "--center", "Flag to calculate the innermost shell only", "", ArgumentHandler::BOOL, &_options._center });
	add({ "--center", "Flag to calculate the outermost shell only", "", ArgumentHandler::BOOL, &_options._edge });
	add({ "--sum-shell", "Flag to sum all of the shells together", "", ArgumentHandler::BOOL, &_options._sumShell });
#ifdef Z_AXIS_ONLY
	add({ "--field", "Flag to sum all of the shells together", "", ArgumentHandler::BOOL, &_options._field });
#endif
	add({ "--gnuplot", "Flag for generating gnuplot files", "", ArgumentHandler::BOOL, &_options._gnuplot });
	add({ "--gnuonly", "Flag for generating only gnuplot files", "", ArgumentHandler::BOOL, &_options._gnuonly });
	add({ "--nonorm", "Flag to not normalize data to 1", "", ArgumentHandler::BOOL, &_options._normalize });
	add({ "(--)help/h/?", "Shows this page", "", ArgumentHandler::CUSTOM, (double*)nullptr });

	if (argc == 1)
	{
		for (auto& a : argumentHandlers) cout << a._option << '\t' << a._help << endl;
		exit(EXIT_SUCCESS);
	}

	for (auto i = 1; i < argc; i++)
	{
		if(strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-help") == 0 || strcmp(argv[i], "help") == 0
		|| strcmp(argv[i], "--h") == 0 || strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "h") == 0
		|| strcmp(argv[i], "--?") == 0 || strcmp(argv[i], "-?") == 0 || strcmp(argv[i], "?") == 0)
		{
			for (auto& a : argumentHandlers) cout << a._option << '\t' << a._help << endl;
			exit(EXIT_SUCCESS);
		}
	}
	
	for (auto i = 1; i < argc; i++)
	{
		if (i >= argc) break;
		if(strcmp(argv[i], "-r") == 0)
		{
			ArgumentHandler& a = argumentHandlers[2];
			try
			{
				double tr = stod(argv[i + 1]);
				_options._r = tr;
			}
			catch (const invalid_argument&)
			{
				cout << "Invalid " << a._usage << endl;
				return EXIT_FAILURE;
			}
			catch (const out_of_range&)
			{
				cout << "Argument out of range for " << a._option << endl;
				return EXIT_FAILURE;
			}
			catch (const exception& ex)
			{
				cout << ex.what() << endl;
				return EXIT_FAILURE;
			}
			catch (...) { }
		}
		if (strcmp(argv[i], "-s") == 0)
		{
			ArgumentHandler& a = argumentHandlers[3];
			try
			{
				double ts = stod(argv[i + 1]);
				_options._stepsize = ts;
			}
			catch (const invalid_argument&)
			{
				cout << "Invalid " << a._usage << endl;
				return EXIT_FAILURE;
			}
			catch (const out_of_range&)
			{
				cout << "Argument out of range for " << a._option << endl;
				return EXIT_FAILURE;
			}
			catch (const exception& ex)
			{
				cout << ex.what() << endl;
				return EXIT_FAILURE;
			}
			catch (...) {}
		}
	}
	
	for (auto i = 1; i < argc; i++)
	{
		bool knownOption = false;
		for (auto& a : argumentHandlers)
		{
			if (i >= argc) break;
			if (strcmp(argv[i], a._option.c_str()) == 0)
			{
				knownOption = true;
				try
				{
					if (a._type == ArgumentHandler::CUSTOM)
					{
						if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "-qr") == 0)
						{
							double factor = 1.0;
							if(strcmp(argv[i], "-qr") == 0) factor = _options._r; 
							int beginI = i, endI = 0;
							bool good = true;
							while (good)
							{
								i++;
								if (i >= argc)
								{
									endI = i;
									good = false;
									break;
								}
								for (auto& b : argumentHandlers)
								{
									if (strcmp(argv[i], b._option.c_str()) == 0)
									{
										endI = i;
										good = false;
										break;
									}
								}
								if (!good) break;

								unsigned steps = TO_UINT(_options._r / _options._stepsize);
								
#ifdef Z_AXIS_ONLY
								vector<Complex> ctemp = vector<Complex>(steps * 2 + 1);
								_zAxisShells.push_back(make_pair(stod(argv[i]) / factor, move(ctemp)));
#else
								vector<SphereShell<long>> temp = vector<SphereShell<long>>(TO_UINT(floor((double)steps / 2.0) + 1));

								for (auto j = 0u; j < temp.size(); j++)
								{
									temp[j]._start = 2 * j;
									temp[j]._end = temp[j]._start + 2;
								}

								_shells.push_back(make_pair(stod(argv[i]) / factor, move(temp)));
#endif
							}
							if (endI - beginI == 1)
							{
								throw invalid_argument("");
							}
						}
						i--;
					}
					if (a._type != ArgumentHandler::BOOL && a._type != ArgumentHandler::CUSTOM) i++;
					if (i >= argc) throw invalid_argument("");
					switch (a._type)
					{
					case ArgumentHandler::STRING:
						(*a._data._str) = string(argv[i]);
						break;
					case ArgumentHandler::BOOL:
						(*a._data._b) = !(*a._data._b);
						break;
					case ArgumentHandler::INT:
						(*a._data._i) = stoi(argv[i]);
						break;
					case ArgumentHandler::DOUBLE:
						(*a._data._d) = stod(argv[i]);
						break;
					case ArgumentHandler::CUSTOM:
						break;
					}
					if (!a._valid()) throw invalid_argument("");
				}
				catch (const invalid_argument&)
				{
					cout << "Invalid " << a._usage << endl;
					return EXIT_FAILURE;
				}
				catch (const out_of_range&)
				{
					cout << "Argument out of range for " << a._option << endl;
					return EXIT_FAILURE;
				}
				catch (const exception& ex)
				{
					cout << ex.what() << endl;
					return EXIT_FAILURE;
				}
				catch (...) { }
				break;
			}
		}
		if (!knownOption)
		{
			cout << "Invalid option: \"" << argv[i] << ".\" Type help for list of options." << endl;
			return EXIT_FAILURE;
		}
	}
	
	_options.update();
	if (!(_options._iVA || _options._shell || _options._halfShell || _options._iVqR || _options._gnuplot))
	{
		cout << "No data is being written to files. Use either --shell or --intensity-vs-angle to generate useable data!" << endl;
		return EXIT_FAILURE;
	}
	
	if(_options._center && _options._edge)
	{
		cout << "Cannot use --center and --edge at the same time!" << endl;
		return EXIT_FAILURE;
	}
	
#ifdef Z_AXIS_ONLY
	if((_options._shell || _options._halfShell) && _zAxisShells.size() == 0)
#else
	if((_options._shell || _options._halfShell) && _shells.size() == 0)
#endif
	{
		cout << "Cannot calculate shells without any given values for q." << endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

void describeRun()
{
	double realm = _options._realm;
	double kappa = _options._kappa;
	double k = _options._k;
	double r = _options._r;
	double sqrtf = ((realm * realm - kappa * kappa - 1) * (realm * realm - kappa * kappa - 1) + 4 * realm * realm * kappa * kappa);
	sqrtf = sqrtf / ((realm * realm - kappa * kappa + 2) * (realm * realm - kappa * kappa + 2) + 4 * realm * realm * kappa * kappa);
	sqrtf = sqrt(sqrtf);
	double rhoprime = 2 * k * r * sqrtf;
	double rho = 2 * k * r * sqrt((realm - 1) * (realm - 1) + kappa * kappa);
	double kKr = kappa * k * r;
	cout << "rhoprime = " << rhoprime << endl;
	cout << "rho = " << rho << endl;
	cout << "kKr = " << kKr << endl;
	
	//Calculate if the wavelength is approximately a factor of pi
	stringstream tempss;
	if(abs((pi / _options._lambda) - round(pi / _options._lambda)) < 0.0000000000001)
	{
		tempss << "π/" << round(pi / _options._lambda);
	}
	else
	{
		tempss << _options._lambda;
	}
	string outWavelength = tempss.str();
	
	//Write general information about the experiement

	cout << "Output prefix: " << _options._file.c_str() << endl;
	cout << "Number of parallel threads: " << _options._parallelthreads << endl;
	cout << "Radius: " << _options._r << endl;
	cout << "Step size: " << _options._stepsize << endl;
	cout << "Theta steps: " << _options._thetastep << endl;
	cout << "Wavelength: " << outWavelength << endl;
	cout << "Size parameter: " << _options._x << endl;
	cout << "Realm: " << _options._realm << endl;
	cout << "Kappa: " << _options._kappa << endl;

#ifdef Z_AXIS_ONLY
	if (_zAxisShells.size() > 0)
	{
		cout << "Testing q values of: ";
		auto p = cout.precision();
		cout << setprecision(10);
		for (auto d : _zAxisShells) cout << d.first << " ";
		cout << endl;
		cout << "Testing qR values of: ";
		for (auto d: _zAxisShells) cout << d.first * _options._r << " ";
		cout << setprecision(p);
		cout << endl;
	}
#else
	if (_shells.size() > 0)
	{
		cout << "Testing q values of: ";
		auto p = cout.precision();
		cout << setprecision(10);
		for (auto d : _shells) cout << d.first << " ";
		cout << endl;
		cout << "Testing qR values of: ";
		for (auto d: _shells) cout << d.first * _options._r << " ";
		cout << setprecision(p);
		cout << endl;
	}
#endif
	
	string infoString = "Calculating fields";

	if(_options._gnuonly)
	{
		cout << "Generating gnuplot files only..." << endl;
		return;
	}
	
	if (_options._iVA || _options._iVqR) infoString.append(" and intensity vs scattered angle");
	if (_options._shell || _options._halfShell) infoString.append(" and shells");
	
#ifdef Z_AXIS_ONLY
	infoString.append(" on the z-axis");
#endif

	cout << infoString.append("...") << endl;
}

void writeDataToFile()
{
	if (_options._iVA)
	{
		//Complex factor = ((_options._m * _options._m) - 1.0) * pow(_options._stepsize, 3.0) * _options._k * _options._k / TO_DBL(4.0L * pi);
		//double amax = abs(_partialSums[0] * factor) * abs(_partialSums[0] * factor);
		double amax = abs(_partialSums[0]) * abs(_partialSums[0]);
		ofstream out(filename(0, DataType::INTENSITY_VS_ANGLE, COMMA_SEPARATED), ios::binary);
		for (auto i = 0u; i < _partialSums.size(); i++)
		{
			/*Complex r = _partialSums[i] * factor;
			double a = abs(r) * abs(r);*/
			double a = abs(_partialSums[i]) * abs(_partialSums[i]);

			double theta = TO_DBL(TO_LDBL(i) * (pi / TO_LDBL(_options._thetastep)));

			if(_options._normalize)
			{
				out << (theta * 180.0 / pi) << outputDelimiter << a / amax << '\n';
			}
			else
			{
				out << (theta * 180.0 / pi) << outputDelimiter << a << '\n';
			}
		}
		out.close();
	}
	
	if (_options._iVqR)
	{
		double amax = abs(_partialSums[0]) * abs(_partialSums[0]);
		ofstream out(filename(0, DataType::INTENSITY_VS_QR, COMMA_SEPARATED), ios::binary);
		for (auto i = 0u; i < _partialSums.size(); i++)
		{
			double a = abs(_partialSums[i]) * abs(_partialSums[i]);

			double theta = TO_DBL(TO_LDBL(i) * (pi / TO_LDBL(_options._thetastep)));

			if(_options._normalize)
			{
				out << (2.0 * _options._k * _options._r * sin(theta / 2.0)) << outputDelimiter << a / amax << '\n';
			}
			else
			{
				out << (2.0 * _options._k * _options._r * sin(theta / 2.0)) << outputDelimiter << a << '\n';
			}
		}
		out.close();
	}
	
	_partialSums.clear();

	if (_options._shell || _options._halfShell)
	{
#ifndef Z_AXIS_ONLY
		
		for (auto& qs : _shells)
		{
			for (auto& s : qs.second)
			{
				double negFactor = s._nNegativePoints;
				double posFactor = s._nPositivePoints;
				s._negativeSum = Complex(s._negativeSum.real() / negFactor, s._negativeSum.imag() / negFactor);
				s._positiveSum = Complex(s._positiveSum.real() / posFactor, s._positiveSum.imag() / posFactor);
			}

			if(_options._halfShell)
			{
				ofstream out(filename(qs.first, DataType::HALF_SHELL, COMMA_SEPARATED), ios::binary);
				out << DATA_HEADER << endl;
				
				if(_options._sumShell)
				{
					_sumType = SUM_TYPE::BOTH;
					Complex s = Complex(0, 0);
					for(auto& r : qs.second)
					{
						s += r.totalSum();
					}
					out << writeData(out, 0, s, qs) << endl;
				}
				else
				{
					_sumType = SUM_TYPE::NEGATIVE;
					SphereShell<long> s = qs.second[qs.second.size() - 1];

					for (auto i = qs.second.size() - 1; i > 0; i--, s = qs.second[i])
					{
						if (s.intRadius() > (_options._r / _options._stepsize)) continue;
						out << writeData(out, -s.intRadius(), s.totalSum(), qs) << endl;
					}

					_sumType = SUM_TYPE::POSITIVE;
					for (auto& s : qs.second)
					{
						if (s.intRadius() > (_options._r / _options._stepsize)) continue;
						out << writeData(out, s.intRadius(), s.totalSum(), qs) << endl;
					}
				}

				out.close();
			}

			if(_options._shell)
			{
				ofstream out(filename(qs.first, DataType::FULL_SHELL, COMMA_SEPARATED), ios::binary);
				out << DATA_HEADER << endl;
				
				_sumType = SUM_TYPE::BOTH;
				
				if(_options._sumShell)
				{
					Complex s = Complex(0, 0);
					for(auto& r : qs.second)
					{
						s += r.totalSum();
					}
					out << writeData(out, 0, s, qs) << endl;
				}
				else
				{
					for (auto& s : qs.second)
					{
						if (s.intRadius() > (_options._r / _options._stepsize)) continue;
						out << writeData(out, s.intRadius(), s.totalSum(), qs) << endl;
					}
				}

				out.close();
			}
		}
	
#else
	
		if (_options._shell)
		{
			for (auto& qs : _zAxisShells)
			{
				ofstream out(filename(qs.first, DataType::FULL_SHELL, COMMA_SEPARATED), ios::binary);
				out << DATA_HEADER << endl;
				
				if(_options._sumShell)
				{
					Complex s = Complex(0, 0);
					for (auto& r : qs.second)
					{
						s += r;
					}
					out << writeData(out, 0, s, qs) << endl;
				}
				else
				{
					for (auto i = 1u; i < qs.second.size() / 2 + 1; i++)
					{
						Complex s = qs.second[i - 1] + qs.second[qs.second.size() - i];
						out << writeData(out, i, s, qs) << endl;
					}
				}

				out.close();
			}
		}
	
		if(_options._halfShell)
		{
			for(auto a = 0u; a < _zAxisShells.size(); a++)
			{
				auto& qs = _zAxisShells[a];
				ofstream out(filename(qs.first, DataType::HALF_SHELL, COMMA_SEPARATED), ios::binary);
				out << DATA_HEADER << endl;
				
				if(_options._sumShell)
				{
					Complex s = Complex(0, 0);
					for (auto& r : qs.second)
					{
						s += r;
					}
					out << writeData(out, 0, s, qs) << endl;
				}
				else
				{
					int i = -TO_INT(_options._r / _options._stepsize);
					for (auto& s : qs.second)
					{
						if(i == 0) i++;
						out << writeData(out, i, s, qs) << endl;
						i++;
					}
				}
				
				out.close();
			}
		}

#endif

	}
}

void writeGnuplotFiles()
{
	stringstream tempss;
	if(abs((pi / _options._lambda) - round(pi / _options._lambda)) < 0.0000000000001) tempss << "π/" << round(pi / _options._lambda);
	else tempss << _options._lambda;
	string outWavelength = tempss.str();
	
	if (_options._iVA)
	{
		ofstream gnu(filename(0, DataType::INTENSITY_VS_ANGLE, PLOT), ios::binary);

		gnu << "set title 'R=" << _options._r << " μm stepsize=" << _options._stepsize << " μm λ=" << outWavelength << " m=" << _options._m.real() << "'" << endl;
		gnu << "set ylabel 'I(θ)/I(0)'" << endl;
		gnu << "set xlabel 'θ'" << endl;
		gnu << "unset logscale" << endl;
		gnu << "set logscale y" << endl;
		gnu << "set datafile sep '" << outputDelimiter << "'" << endl;
		gnu << "plot \"" << filename(0, DataType::INTENSITY_VS_ANGLE, COMMA_SEPARATED, false) << "\" using 1:2 with lines title 'perpendicular'" << endl;

		gnu.close();
	}
	
	if (_options._iVqR)
	{
		ofstream gnu(filename(0, DataType::INTENSITY_VS_QR, PLOT), ios::binary);

		gnu << "set title 'R=" << _options._r << " μm stepsize=" << _options._stepsize << " μm λ=" << outWavelength << " m=" << _options._m.real() << "'" << endl;
		gnu << "set ylabel 'I(qR)/I(0)'" << endl;
		gnu << "set xlabel 'qR'" << endl;
		gnu << "unset logscale" << endl;
		gnu << "set logscale y" << endl;
		gnu << "set logscale x" << endl;
		gnu << "set datafile sep '" << outputDelimiter << "'" << endl;
		gnu << "plot \"" << filename(0, DataType::INTENSITY_VS_QR, COMMA_SEPARATED, false) << "\" using 1:2 with lines title 'perpendicular'" << endl;

		gnu.close();
	}

	if (_options._shell || _options._halfShell)
	{	
#ifndef Z_AXIS_ONLY
		string ylabel = "E_{shell}(r)";
		if (_options._normalize) ylabel.append(" normalized to 1");
#else
		string ylabel = "E(r)";
		if (_options._normalize) ylabel.append(" normalized to 1");
#endif

#ifndef Z_AXIS_ONLY
		for (auto& qs : _shells)
#else
		for (auto& qs : _zAxisShells)
#endif
		{
			if(_options._halfShell)
			{
				ofstream gnu(filename(qs.first, DataType::HALF_SHELL, VECTORS + PLOT), ios::binary);

				gnu << "set title '" << half_gnu_title << " R=" << _options._r << " μm λ=" << outWavelength << " m=" << _options._m.real();
#ifndef Z_AXIS_ONLY
				gnu << " R_{shell}=" << 2.0 * _options._stepsize << " μm";
#endif
				gnu << " qR=" << qs.first * _options._r << "'" << endl;
				gnu << "set ylabel '" << ylabel << "'" << endl;
				gnu << "set xlabel 'r'" << endl;
				gnu << "unset logscale" << endl;
				gnu << "set datafile sep '" << outputDelimiter << "'" << endl;
				gnu << "plot \"" << filename(qs.first, DataType::HALF_SHELL, COMMA_SEPARATED, false) << "\" every 5 using 6:(0):7:8 with vectors title 'qR=" << (qs.first * _options._r) << "'" << endl;

				gnu.close();

				gnu.open(filename(qs.first, DataType::HALF_SHELL, LINES + PLOT), ios::binary);

				gnu << "set title '" << half_gnu_title << " R=" << _options._r << " μm λ=" << outWavelength << " m=" << _options._m.real();
#ifndef Z_AXIS_ONLY
				gnu << " R_{shell}=" << 2.0 * _options._stepsize << " μm";
#endif
				gnu << " qR=" << qs.first * _options._r << "'" << endl;
				gnu << "set ylabel '" << ylabel << "'" << endl;
				gnu << "set xlabel 'r'" << endl;
				gnu << "unset logscale" << endl;
				gnu << "set datafile sep '" << outputDelimiter << "'" << endl;
				gnu << "plot ";
				gnu << "\"" << filename(qs.first, DataType::HALF_SHELL, COMMA_SEPARATED, false) << "\" using 6:7 with lines title 'real', \\" << endl;
				gnu << "\"" << filename(qs.first, DataType::HALF_SHELL, COMMA_SEPARATED, false) << "\" using 6:8 with lines title 'imaginary'" << endl;

				gnu.close();
			}
			
			if(_options._shell)
			{
				ofstream gnu(filename(qs.first, DataType::FULL_SHELL, VECTORS + PLOT), ios::binary);

				gnu << "set title '" << gnu_title << " R=" << _options._r << " μm λ=" << outWavelength << " m=" << _options._m.real();
#ifndef Z_AXIS_ONLY
				gnu << " R_{shell}=" << 2.0 * _options._stepsize << " μm";
#endif
				gnu << " qR=" << qs.first * _options._r << "'" << endl;
				gnu << "set ylabel '" << ylabel << "'" << endl;
				gnu << "set xlabel 'r'" << endl;
				gnu << "unset logscale" << endl;
				gnu << "set datafile sep '" << outputDelimiter << "'" << endl;
				gnu << "plot \"" << filename(qs.first, DataType::FULL_SHELL, COMMA_SEPARATED, false) << "\" every 5 using 6:(0):7:8 with vectors title 'qR=" << (qs.first * _options._r) << "'" << endl;

				gnu.close();

				gnu.open(filename(qs.first, DataType::FULL_SHELL, LINES + PLOT), ios::binary);

				gnu << "set title '" << gnu_title << " R=" << _options._r << " μm λ=" << outWavelength << " m=" << _options._m.real();
#ifndef Z_AXIS_ONLY
				gnu << " R_{shell}=" << 2.0 * _options._stepsize << " μm";
#endif
				gnu << " qR=" << qs.first * _options._r << "'" << endl;
				gnu << "set ylabel '" << ylabel << "'" << endl;
				gnu << "set xlabel 'r'" << endl;
				gnu << "unset logscale" << endl;
				gnu << "set datafile sep '" << outputDelimiter << "'" << endl;
				gnu << "plot ";
				gnu << "\"" << filename(qs.first, DataType::FULL_SHELL, COMMA_SEPARATED, false) << "\" using 6:7 with lines title 'real', \\" << endl;
				gnu << "\"" << filename(qs.first, DataType::FULL_SHELL, COMMA_SEPARATED, false) << "\" using 6:8 with lines title 'imaginary'" << endl;

				gnu.close();
			}
		}
	}
}

void calcAnx(vector<Complex>& Anx, int nmax, double x)
{
	for (int i = nmax - 1; i >= 1; i--)
	{
		Anx[i] = (i + 1.0) / x - 1.0 / (((i + 1.0) / x) + Anx[i + 1]);
	}
}

void calcAnmx(vector<Complex>& Anmx, int nmax, const Complex& mx)
{
	for (int i = nmax - 1; i >= 1; i--)
	{
		Anmx[i] = (i + 1.0) / mx - 1.0 / (((i + 1.0) / mx) + Anmx[i + 1]);
	}
}

void calcBnx(vector<Complex>& Bnx, int nmax, double x)
{
	Bnx[0] = Complex(0.0, 1.0);
	for (int i = 1; i < nmax; i++)
	{
		Bnx[i] = -(i / x) + 1.0 / ((i / x) - Bnx[i - 1]);
	}
}

void calcBnmx(vector<Complex>& Bnmx, int nmax, const Complex& mx)
{
	Bnmx[0] = Complex(0.0, 1.0);
	for (int i = 1; i < nmax; i++)
	{
		Bnmx[i] = -((i * 1.0) / mx) + 1.0 / (((i * 1.0) / mx) - Bnmx[i - 1]);
	}
}

void calcPsinx(vector<Complex>& Anx, vector<Complex>& psinx, int nmax, double x)
{
	psinx[0] = Complex(sin(x), 0);
	for (int i = 1; i < nmax; i++)
	{
		psinx[i] = psinx[i - 1] / (Anx[i] + (i / x));
	}
}

void calcPsinmx(vector<Complex>& Anmx, vector<Complex>& psinmx, int nmax, const Complex& mx)
{
	psinmx[0] = Complex(sin(real(mx)) * cosh(imag(mx)), cos(real(mx)) * sinh(imag(mx)));
	for (int i = 1; i < nmax; i++)
	{
		psinmx[i] = psinmx[i - 1] / (Anmx[i] + ((i * 1.0) / mx));
	}
}

void calcXix(vector<Complex>& Bnx, vector<Complex>& xix, int nmax, double x)
{
	xix[0] = Complex(sin(x), -cos(x));
	for (int i = 1; i < nmax; i++)
	{
		xix[i] = xix[i - 1] / (Bnx[i] + (i / x));
	}
}

void calcXinmx(vector<Complex>& Bnmx, vector<Complex>& xinmx, int nmax, const Complex& mx)
{
	xinmx[0] = Complex(sin(real(mx)) * cosh(imag(mx)) - sin(real(mx)) * sinh(imag(mx)), cos(real(mx)) * sinh(imag(mx)) - cos(real(mx)) * cosh(imag(mx)));
	for (int i = 1; i < nmax; i++)
	{
		xinmx[i] = xinmx[i - 1] / (Bnmx[i] + ((i * 1.0) / mx));
	}
}

Complex En(int n)
{
	Complex temp = Complex(0.0, 1.0);
	temp = pow(temp, n);

	double temp1 = ((2.0 * 1.0 * n + 1) / (n * 1.0 * (n * 1.0 + 1)));
	return (temp * temp1);
}

Complex Enr(int n)
{
	Complex temp = Complex(0.0, 1.0);
	temp = pow(temp, n);

	double temp1 = ((2.0 * 1.0 * n + 1));
	return (temp * temp1);
}

void calcCartesianFields(int start, int length, int steps)
{
#ifndef Z_AXIS_ONLY
	int end = start + length;
#endif
	unsigned nmax = _options._nmax;

	vector<Complex> psinx(nmax + 1, Complex());
	vector<Complex> psinmx(nmax + 1, Complex());

	vector<Complex> xix(nmax + 1, Complex());
	vector<Complex> xinmx(nmax + 1, Complex());

	vector<Complex> Anx(nmax + 1, Complex());
	vector<Complex> Anmx(nmax + 1, Complex());

	vector<Complex> psinmrk(nmax + 1, Complex());
	vector<Complex> Anmrk(nmax + 1, Complex());

	vector<Complex> Bnx(nmax + 1, Complex());
	vector<Complex> Bnmx(nmax + 1, Complex());

	vector<Complex> cn(nmax + 1, Complex());
	vector<Complex> dn(nmax + 1, Complex());

	calcAnx(Anx, nmax, _options._x);
	calcAnmx(Anmx, nmax, _options._mx);
	calcBnx(Bnx, nmax, _options._x);
	calcBnmx(Bnmx, nmax, _options._mx);
	calcPsinmx(Anmx, psinmx, nmax, _options._mx);
	calcPsinx(Anx, psinx, nmax, _options._x);
	calcXix(Bnx, xix, nmax, _options._x);
	calcXinmx(Bnmx, xinmx, nmax, _options._mx);

	Complex temp1;
	Complex temp2;
	Complex temp3;

	for (int i = 1; i <= _options._xstop; i++)
	{
		temp1 = Bnx[i] - Anx[i];
		temp2 = (Bnx[i] / _options._m) - Anmx[i];
		temp3 = psinx[i] / psinmx[i];
		cn[i] = temp3 * (temp1 / temp2);

		temp2 = (Bnx[i]) - (Anmx[i] / _options._m);
		dn[i] = temp3 * (temp1 / temp2);
	}
	
	double taun = 0;
	double pii_minus_2 = 0;
	double pii_minus_1 = 1;
	double piin = 1;

	double tempa = 0;
	double tempb = 0;

	double currentr = 0;
	double currenttheta = 0;
	double currentphi = 0;
	double cosphi = 0;
	double costheta = 0;
	double sinphi = 0;
	double sintheta = 0;

	Complex mkr, Er, Etheta, Ephi, Ex;// , Ey, Ez;

	vector<Complex> partialSums;
	if(_options._iVA || _options._iVqR) partialSums = vector<Complex>(_options._thetastep + 1, Complex());

#ifndef Z_AXIS_ONLY
	vector<pair<double, vector<SphereShell<long>>>> partialShells;
	if (_options._shell || _options._halfShell)
	{
		partialShells = vector<pair<double, vector<SphereShell<long>>>>(_shells.size());

		for (auto i = 0u; i < partialShells.size(); i++)
		{
			partialShells[i].first = _shells[i].first;
			partialShells[i].second = _shells[i].second;
		}
	}
#endif

	try
	{
#ifndef Z_AXIS_ONLY
		for (auto a = start; a < end; a++)
		{
			for (auto b = 0; b <= steps; b++)
			{
#else
				int a = 0, b = 0;
#endif
				for (auto k = -steps; k <= steps; k++)
				{
					if (a == 0 && b == 0 && k == 0) k++;

					double xxx = TO_DBL(a) * _options._stepsize;
					double yyy = TO_DBL(b) * _options._stepsize;
					double zzz = TO_DBL(k) * _options._stepsize;

					if (sqrt(a * a + b * b + k * k) <= steps)
					{
						Er = Complex(0, 0);
						Etheta = Complex(0, 0);
						Ephi = Complex(0, 0);
						currentr = sqrt((xxx * xxx) + (yyy * yyy) + (zzz * zzz));
						currenttheta = atan2(sqrt(xxx * xxx + yyy * yyy), zzz);
						currentphi = atan2(yyy, xxx);

						if (currentphi < 0) currentphi = currentphi + pi2;
						if (currenttheta < 0) currenttheta = currenttheta + pi2;

						cosphi = cos(currentphi);
						costheta = cos(currenttheta);
						sinphi = sin(currentphi);
						sintheta = sin(currenttheta);
						//---------------------------------------------------------------------//
						/*
						and this is where the actually internal field is calculated
						*/
						//---------------------------------------------------------------------//
						mkr = _options._m * _options._k * currentr;

						calcAnmx(Anmrk, nmax, mkr);
						calcPsinmx(Anmrk, psinmrk, nmax, mkr);

						taun = costheta;
						pii_minus_2 = 0;
						pii_minus_1 = 1;
						piin = 1;

						for (int i = 1; i <= _options._xstop; i++)
						{
							if (2 <= i)
							{
								tempa = (((2.0 * i) - 1.0) / (i - 1.0));
								tempb = ((i * 1.0) / (TO_DBL(i) - 1.0));
								piin = tempa * costheta * pii_minus_1 - tempb * pii_minus_2;
								taun = i * costheta * piin - (i + 1.0) * pii_minus_1;
								pii_minus_2 = pii_minus_1;
								pii_minus_1 = piin;
							}

							Er = Er - cosphi * sintheta * imaginary_unit * Enr(i) * dn[i] * piin * psinmrk[i];
							Etheta = Etheta + En(i) * psinmrk[i] * (cn[i] * piin - imaginary_unit * dn[i] * taun * Anmrk[i]);
							Ephi = Ephi + En(i) * psinmrk[i] * (imaginary_unit * dn[i] * piin * Anmrk[i] - cn[i] * taun);
						}

						Er = Er / ((mkr)* (mkr));
						Etheta = Etheta * ((cosphi) / (mkr));
						Ephi = Ephi * ((sinphi) / (mkr));

						// these are the final fields we are looking for back in cartesian //
						Ex = Er * sintheta * cosphi + Etheta * costheta * cosphi - Ephi * sinphi;
						//Ey = Er * sintheta * sinphi + Etheta * costheta * sinphi + Ephi * cosphi;
						//Ez = Er * costheta - Etheta * sintheta;
						
						pair<Point<long>, Complex> f, f1, f2, f3;
						f.first._x = a;
						f.first._y = b;
						f.first._z = k;
						f.second = Ex;

						vector<pair<Point<long>, Complex>> tempFields;
						tempFields.push_back(f);

						if (b == 0 && a != 0)
						{
							f1.first._x = -a;
							f1.first._y = b;
							f1.first._z = k;
							f1.second = Ex;
							tempFields.push_back(f1);
						}

						if (a == 0 && b != 0)
						{
							f2.first._x = a;
							f2.first._y = -b;
							f2.first._z = k;
							f2.second = Ex;
							tempFields.push_back(f2);
						}

						if (b != 0 && a != 0)
						{
							f1.first._x = -a;
							f1.first._y = b;
							f1.first._z = k;
							f1.second = Ex;
							tempFields.push_back(f1);

							f2.first._x = a;
							f2.first._y = -b;
							f2.first._z = k;
							f2.second = Ex;
							tempFields.push_back(f2);

							f3.first._x = -a;
							f3.first._y = -b;
							f3.first._z = k;
							f3.second = Ex;
							tempFields.push_back(f3);
						}

						if (_options._iVA || _options._iVqR)
						{
							for (auto ts = 0; ts <= _options._thetastep; ts++)
							{
								double theta = TO_DBL(ts) * (pi / TO_DBL(_options._thetastep));
								double ksz = _options._k * cos(theta);
								double ksy = _options._k * sin(theta);

								double kiz = _options._k;

								double qz = kiz - ksz;
								double qy = -ksy;
								double qx = 0;

								Complex sum = Complex(0, 0);
								double qdotr = 0;
								double kidotr = 0;

								for (auto& field : tempFields)
								{
									auto po = field.first.Convert<double>(_options._stepsize);
									qdotr = qx * po._x + qy * po._y + qz * po._z;
									kidotr = kiz * po._z;

									sum += field.second * exp(imaginary_unit * qdotr) * exp(Complex(0, -1.0) * kidotr);
								}

								partialSums[ts] += sum;
							}
						}

#ifndef Z_AXIS_ONLY
						//------- Where the shells are calculated --------------//

						if (_options._shell || _options._halfShell)
						{
							double radius = sqrt(a * a + b * b + k * k);

							for (auto& psi : partialShells)
							{
								double theta = 2.0 * asin(psi.first / (2.0 * _options._k));
								
								for (auto& psj : psi.second)
								{
									if (radius > psj._start && radius <= psj._end)
									{

										double ksz = _options._k * cos(theta);
										double ksy = _options._k * sin(theta);

										double kiz = _options._k;

										double qz = kiz - ksz;
										double qy = -ksy;
										double qx = 0;

										double qdotr = 0;
										double kidotr = 0;

										for (auto& field : tempFields)
										{
											auto po = field.first.Convert<double>(_options._stepsize);
											qdotr = qx * po._x + qy * po._y + qz * po._z;
											kidotr = kiz * po._z;
											
											if(_options._field)
											{
												if (k < 0)
												{
													psj._negativeSum += field.second;
													psj._nNegativePoints++;
												}
												else
												{
													psj._positiveSum += field.second;
													psj._nPositivePoints++;
												}
											}
											else
											{
												Complex exponential = exp(imaginary_unit * qdotr) * exp(Complex(0, -1.0) * kidotr);

												if (k < 0)
												{
													psj._negativeSum += field.second * exponential;
													psj._nNegativePoints++;
												}
												else
												{
													psj._positiveSum += field.second * exponential;
													psj._nPositivePoints++;
												}
											}
										}
									}
								}
							}
						}
						
						//-------------------------------------------------------//

						tempFields.clear();
#else
						if(_options._shell || _options._halfShell)
						{
							for (auto& psi : _zAxisShells)
							{
								double theta = 2.0 * asin(psi.first / (2.0 * _options._k));
								
								double ksz = _options._k * cos(theta);
								double ksy = _options._k * sin(theta);
								
								double kiz = _options._k;

								double qz = kiz - ksz;
								double qy = -ksy;
								double qx = 0;

								double qdotr = 0;
								double kidotr = 0;
								
								auto po = f.first.Convert<double>(_options._stepsize);
								qdotr = qx * po._x + qy * po._y + qz * po._z;
								kidotr = kiz * po._z;
								
								if(_options._field)
								{
									psi.second[k + steps] = f.second;
								}
								else
								{
									psi.second[k + steps] = f.second * exp(imaginary_unit * qdotr) * exp(Complex(0, -1.0) * kidotr);
								}
							}
						}
#endif
					}
				}
#ifndef Z_AXIS_ONLY
			}
		}
#endif
	}
	catch (bad_alloc& ba)
	{
		cout << endl << "A bad_alloc exception has been thrown while calculating the fields" << endl << endl;
		cout << ba.what() << endl << endl;
		exit(EXIT_FAILURE);
	}
	catch (exception& ex)
	{
		cout << endl << "An exception occurred while calculating the fields" << endl << endl;
		cout << ex.what() << endl << endl;
		exit(EXIT_FAILURE);
	}
	catch (...)
	{
		exit(EXIT_FAILURE);
	}
#ifndef Z_AXIS_ONLY
	if (_options._iVA || _options._iVqR)
	{
		try
		{
			_fieldLock.lock();
			for (auto i = 0; i <= _options._thetastep; i++)
			{
				_partialSums[i] += partialSums[i];
			}
			_fieldLock.unlock();
			partialSums.clear();
		}
		catch (exception& ex)
		{
			cout << endl << "An exception occurred when adding the partial sums" << endl << endl;
			cout << ex.what() << endl << endl;
			exit(EXIT_FAILURE);
		}
		catch (...)
		{
			exit(EXIT_FAILURE);
		}
	}
	if (_options._shell || _options._halfShell)
	{
		try
		{
			_fieldLock.lock();
			for (auto i = 0u; i < partialShells.size(); i++)
			{
				for (auto j = 0u; j < partialShells[i].second.size(); j++)
				{
					_shells[i].second[j]._negativeSum = partialShells[i].second[j]._negativeSum;
					_shells[i].second[j]._positiveSum = partialShells[i].second[j]._positiveSum;
					_shells[i].second[j]._nNegativePoints = partialShells[i].second[j]._nNegativePoints;
					_shells[i].second[j]._nPositivePoints = partialShells[i].second[j]._nPositivePoints;
				}
			}
			_fieldLock.unlock();
			partialShells.clear();
			partialShells.shrink_to_fit();
		}
		catch (exception& ex)
		{
			cout << endl << "An exception occurred when adding the shell sums" << endl << endl;
			cout << ex.what() << endl << endl;
			exit(EXIT_FAILURE);
		}
		catch (...)
		{
			exit(EXIT_FAILURE);
		}
	}
#endif
}

void tryMakeDir(string dir, int o, bool verbose)
{
#ifdef _WIN32
	dir.erase(std::remove(dir.begin(), dir.end(), '\\'), dir.end());
	replace(dir.begin(), dir.end(), '/', '\\');
	string cmd = "mkdir \"" + dir + "\"";
#else
	string cmd = "mkdir -p \"" + dir + "\"";
#endif
	if (system(cmd.c_str()) == -1 && verbose)
	{
		cout << "Error creating directory: " << dir << endl;
	}
}

string filename(double q, DataType type, string extension, bool inclDir)
{
#ifdef Z_AXIS_ONLY
	string s = "Z-axis/";
#else
	string s;
#endif
	string dir;
	if (inclDir)
	{
		switch (type)
		{
		case INTENSITY_VS_ANGLE:
		case INTENSITY_VS_QR:
			dir = "out/" + s + to_string(_options._m.real()) + "/";
			break;
		case FULL_SHELL:
			dir = "out/" + s + "Full Shell/" + to_string(_options._m.real()) + "/";
			break;
		case HALF_SHELL:
			dir = "out/" + s + "Half Shell/" + to_string(_options._m.real()) + "/";
			break;
		}
	}
	if(_options._field)
	{
		switch (type)
		{
		case INTENSITY_VS_ANGLE:
			return dir + _options._file + "-intensity_vs_angle" + extension;
		case INTENSITY_VS_QR:
			return dir + _options._file + "-intensity_vs_qR" + extension;
		case FULL_SHELL:
			return dir + _options._file + "-shell-field" + extension;
		case HALF_SHELL:
			return dir + _options._file + "-half-shell-field" + extension;
		}
	}
	else
	{
		switch (type)
		{
		case INTENSITY_VS_ANGLE:
			return dir + _options._file + "-intensity_vs_angle" + extension;
		case INTENSITY_VS_QR:
			return dir + _options._file + "-intensity_vs_qR" + extension;
		case FULL_SHELL:
			return dir + _options._file + "-qR_" + leadingZeros(q) + to_string(q * _options._r) + "-shells" + extension;
		case HALF_SHELL:
			return dir + _options._file + "-qR_" + leadingZeros(q) + to_string(q * _options._r) + "-half-shells" + extension;
		}
	}
	return "this-file-should-not-exist.txt";
}

string leadingZeros(double q)
{
#ifdef Z_AXIS_ONLY
	double maxqR = max_element(_zAxisShells.begin(), _zAxisShells.end(), [](pair<double, vector<Complex>>& a, pair<double, vector<Complex>>& b) { return a.first < b.first; })->first * _options._r;
#else
	double maxqR = max_element(_shells.begin(), _shells.end(), [](pair<double, vector<SphereShell<long>>>& a, pair<double, vector<SphereShell<long>>>& b) { return a.first < b.first; })->first * _options._r;
#endif
	long ceilMaxqR = TO_INT(ceil(maxqR));
	long start = 1000000000;
	for (; ceilMaxqR % start == ceilMaxqR && ceilMaxqR != 0; start /= 10);
	if (ceilMaxqR == 0) start = 0;
	long n = 0;
	if (start > 0) n = TO_INT(log10(start));

	string zeros;
	for (int i = 1; i < n + 1; i++)
	{
		if ((q * _options._r) < pow(10, i))
		{
			zeros.append("0");
		}
	}

	return zeros;
}
