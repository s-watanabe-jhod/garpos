# Makefile for GARPOS Fortran ray-tracing library
FC = gfortran
FFLAGS = -shared -fPIC -fopenmp -O3
TARGET = lib_raytrace.so
SOURCES = sub_raytrace.f90 lib_raytrace.f90

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(FC) $(FFLAGS) -o $@ $^

clean:
	rm -f $(TARGET) *.mod
