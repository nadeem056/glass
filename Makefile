# Default target
all: 
	    build

# Build the Go executable
build: 
	    go build -o bin/cloudways cmd/glass.go

# Clean up build artifacts
clean:
	    rm -f bin/cloudway
