package collectors

import (
	"github.com/rs/zerolog/log"
	"github.com/shirou/gopsutil/v4/disk"
)

type DiskCollector struct{}

func (d *DiskCollector) Collector() error {
	diskstat, err := disk.Usage("/")
	if err != nil {
		log.Err(err).Msg("Error getting disk info")
	}
	log.Info().Uint64("total", diskstat.Total).Uint64("free", diskstat.Free).Uint64("used", diskstat.Used).Float64("used-percent", diskstat.UsedPercent).Msg("")
	//log.Info().Msg("Disk collector")
	return nil
}
