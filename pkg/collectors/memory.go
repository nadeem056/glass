package collectors

import (
	"github.com/rs/zerolog/log"
	"github.com/shirou/gopsutil/v4/mem"
)

type MemoryCollector struct{}

func (m *MemoryCollector) Collector() error {
	vmstat, err := mem.VirtualMemory()
	if err != nil {
		log.Err(err).Msg("Error getting memory info")
	}
	log.Info().Uint64("total", vmstat.Total).Uint64("available", vmstat.Available).Uint64("used", vmstat.Used).Uint64("free", vmstat.Free).Float64("used-percent", vmstat.UsedPercent).Msg("")
	//log.Info().Msg("Memory collector")
	return nil
}
