package collectors

type Collector interface {
	Collector() error
}

func RegisterCollectors() []Collector {
	return []Collector{
		&CPUCollector{},
		&MemoryCollector{},
		&DiskCollector{},
		&NetworkCollector{},
	}
}
