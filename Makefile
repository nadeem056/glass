# Default target
all: 
		build

# Build the Go executable
build: 
		echo "Building the Go executable..."
		go build -o bin/cloudways cmd/glass.go

# Clean up build artifacts
clean:
		echo "Cleaning up build artifacts..."
		rm -f bin/cloudway
