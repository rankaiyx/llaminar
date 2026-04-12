/**
 * @file PCIeEnumeration.cpp
 * @brief PCIe link speed/width detection via Linux sysfs
 *
 * Reads current and max link speed/width from:
 *   /sys/bus/pci/devices/<BDF>/current_link_speed
 *   /sys/bus/pci/devices/<BDF>/current_link_width
 *   /sys/bus/pci/devices/<BDF>/max_link_speed
 *   /sys/bus/pci/devices/<BDF>/max_link_width
 */

#include "GPUEnumeration.h"
#include "../utils/Logger.h"
#include <cstdio>
#include <cstring>
#include <string>

namespace llaminar2
{
    namespace pcie_enumeration
    {
        namespace
        {
            // Read a single-line sysfs file into a string, trimming whitespace
            std::string read_sysfs_string(const char *path)
            {
                FILE *f = fopen(path, "r");
                if (!f)
                    return {};
                char buf[128] = {};
                if (!fgets(buf, sizeof(buf), f))
                {
                    fclose(f);
                    return {};
                }
                fclose(f);
                // Trim trailing newline/whitespace
                size_t len = strlen(buf);
                while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r' || buf[len - 1] == ' '))
                    buf[--len] = '\0';
                return std::string(buf);
            }

            // Parse "X.X GT/s" or "X GT/s PCIe" → GT/s as double
            double parse_link_speed(const std::string &s)
            {
                if (s.empty())
                    return 0;
                double speed = 0;
                if (sscanf(s.c_str(), "%lf", &speed) == 1)
                    return speed;
                return 0;
            }

            // Parse width string (just an integer like "16")
            int parse_link_width(const std::string &s)
            {
                if (s.empty())
                    return 0;
                int width = 0;
                if (sscanf(s.c_str(), "%d", &width) == 1)
                    return width;
                return 0;
            }

            // Map GT/s → PCIe generation
            int speed_to_pcie_gen(double gts)
            {
                if (gts >= 64.0)
                    return 6;
                if (gts >= 32.0)
                    return 5;
                if (gts >= 16.0)
                    return 4;
                if (gts >= 8.0)
                    return 3;
                if (gts >= 5.0)
                    return 2;
                if (gts >= 2.5)
                    return 1;
                return 0;
            }
        } // anonymous namespace

        PCIeLinkInfo read_pcie_link_info(int pci_domain, int pci_bus, int pci_device)
        {
            PCIeLinkInfo info;

            char bdf[64];
            snprintf(bdf, sizeof(bdf), "%04x:%02x:%02x.0", pci_domain, pci_bus, pci_device);
            info.pci_address = bdf;

            char path[256];
            const char *base = "/sys/bus/pci/devices";

            // Current link speed
            snprintf(path, sizeof(path), "%s/%s/current_link_speed", base, bdf);
            info.link_speed_gts = parse_link_speed(read_sysfs_string(path));

            // Current link width
            snprintf(path, sizeof(path), "%s/%s/current_link_width", base, bdf);
            info.link_width = parse_link_width(read_sysfs_string(path));

            // Max link speed
            snprintf(path, sizeof(path), "%s/%s/max_link_speed", base, bdf);
            info.max_speed_gts = parse_link_speed(read_sysfs_string(path));

            // Max link width
            snprintf(path, sizeof(path), "%s/%s/max_link_width", base, bdf);
            info.max_width = parse_link_width(read_sysfs_string(path));

            // Derive PCIe generation from current speed
            info.pcie_gen = speed_to_pcie_gen(info.link_speed_gts);

            // Check for degraded link
            info.degraded = (info.link_speed_gts > 0 && info.max_speed_gts > 0 &&
                             (info.link_speed_gts < info.max_speed_gts || info.link_width < info.max_width));

            return info;
        }

    } // namespace pcie_enumeration
} // namespace llaminar2
