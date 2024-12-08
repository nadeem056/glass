package collectors

import (
	"github.com/rs/zerolog/log"
	"github.com/shirou/gopsutil/v4/cpu"
)

type CPUCollector struct {
}

type CPUInformation struct {
	Vendor string  `json:"vendor"`
	Freq   float64 `json:"frequency"`
	Cores  int     `json:"cores"`
	Cache  int     `json:"cache"`
	VCPU   int     `json:"vCPU"`
}

type CPUTimesValues struct {
	Timestamp int64         `json:"timestamp"`
	CPUTimes  cpu.TimesStat `json:"cpu-times"`
}

func (c *CPUCollector) Name() string {
	return "CPU Collector"
}

func (c *CPUCollector) CPUInformation() (CPUInformation, error) {
	cpuInfo, err := cpu.Info()
	if err != nil {
		return CPUInformation{}, err
	}
	Info := CPUInformation{
		Vendor: cpuInfo[0].VendorID,
		Freq:   cpuInfo[0].Mhz,
		Cores:  int(cpuInfo[0].Cores),
		Cache:  int(cpuInfo[0].CacheSize),
		VCPU:   len(cpuInfo),
	}
	return Info, nil
}

func (c *CPUCollector) Collector() error {
	cpuInfo, err := cpu.Info()
	if err != nil {
		log.Error().Err(err).Msg("Error getting CPU info")
	}
	vendor := cpuInfo[0].VendorID
	freq := cpuInfo[0].Mhz
	cores := cpuInfo[0].Cores
	cache := cpuInfo[0].CacheSize
	vCPU, err := cpu.Counts(true)
	if err != nil {
		log.Error().Err(err).Msg("Error getting vCPU count")
	}
	log.Info().Str("vendor", vendor).Float64("freq", freq).Int("cores", int(cores)).Int("cache", int(cache)).Int("vCPU", vCPU).Msg("")
	times, err := cpu.Times(false)
	if err != nil {
		log.Error().Err(err).Msg("Error getting CPU times")
	}
	for _, time := range times {
		log.Info().Str("cpu", time.CPU).Float64("user", time.User).Float64("system", time.System).Float64("idle", time.Idle).Float64("nice", time.Nice).Float64("iowait", time.Iowait).Float64("irq", time.Irq).Float64("softirq", time.Softirq).Float64("steal", time.Steal).Float64("guest", time.Guest).Float64("guest-nice", time.GuestNice).Msg("")
	}
	return nil
}
