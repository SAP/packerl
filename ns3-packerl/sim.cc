#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/ns3-ai-module.h"
#include "ns3/point-to-point-layout-module.h"
#include "ns3/point-to-point-module.h"

#define RYML_SINGLE_HDR_DEFINE_NOW
#include "utils/ryml_all.hpp"

#include "packerl.h"

#include <iostream>
#include <optional>
#include <string>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("PackerlSim");

/**
 * Invoked with yaml parse results; sets the corresponding SimParameters struct member.
 */
void setSimParameter(SimParameters &simParams, const std::string &keyStr, const std::string &valStr)
{
    if (keyStr == "ms_per_step")
    {
        double simStepDuration = std::stod(valStr);
        simParams.simStepDuration = simStepDuration;
    }
    else if (keyStr == "ospfw_ref_value")
    {
        uint32_t ospfwRefValue = static_cast<uint32_t>(std::stoul(valStr));
        simParams.ospfwRefValue = ospfwRefValue;
    }
    else if (keyStr == "packet_size")
    {
        uint32_t packetSize = static_cast<uint32_t>(std::stoul(valStr));
        simParams.packetSize = packetSize;
    }
    else if (keyStr == "use_flow_control")
    {
        simParams.useFlowControl = (valStr == "true" || valStr == "True");
    }
    else if (keyStr == "prob_tcp")
    {
        double probTcp = std::stod(valStr);
        simParams.probTcp = probTcp;
    }
    else if (keyStr == "use_tcp_sack")
    {
        simParams.useTcpSack = (valStr == "true" || valStr == "True");
    }
    else if (keyStr == "probabilistic_routing")
    {
        simParams.probabilisticRouting = (valStr == "true" || valStr == "True");
    }
}

/**
 * Main method of the PackeRL sim.
 * Gets called at the start of every episode, and after termination the episode ends.
 */
int
main(int argc, char* argv[])
{
    std::string configFilePath = "config/default/params.yaml";
    std::string outDir = ".";
    uint32_t run = 0;
    uint32_t memblockKey = 0;

    CommandLine cmd;
    cmd.AddValue("run", "Run index (for setting repeatable seeds)", run);
    cmd.AddValue("memblockKey", "Memblock key for interaction with ns3-ai", memblockKey);
    cmd.AddValue("configFilePath", "Relative path to general config file", configFilePath);
    cmd.AddValue("outDir", "out path for simulation results and profiling", outDir);
    cmd.Parse(argc, argv);

    // parse yaml config file and fill SimParameters struct
    std::ifstream yamlFile(configFilePath); // Replace "example.txt" with your file name
    NS_ASSERT_MSG(yamlFile.is_open(), "couldn't open config file");
    std::string yamlContent((std::istreambuf_iterator<char>(yamlFile)), (std::istreambuf_iterator<char>()));
    ryml::Tree tree = ryml::parse_in_arena(ryml::to_csubstr(yamlContent)); // immutable (csubstr) overload
    ryml::ConstNodeRef root = tree.rootref();
    SimParameters simParameters;
    simParameters.memblockKey = memblockKey;
    for(ryml::ConstNodeRef const& child : root.children())
    {
        std::string childKey(child.key().begin(), child.key().end());
        std::string childVal(child.val().begin(), child.val().end());
        setSimParameter(simParameters, childKey, childVal);
    }

    NS_LOG_INFO("shm sizes: act " << sizeof(PackerlActStruct) << ", env " << sizeof(PackerlEnvStruct) );

    RngSeedManager::SetRun(run);

    // set up env, and signal readiness to python side
    PackerlEnv env(simParameters);
    env.EnvSetterCond()->simReady = true;
    env.SetCompleted();
    NS_LOG_INFO("initialized PackeRL");

    // run the simulator loop
    bool ok = env.runSimLoop();

    // experiment cleanup
    NS_LOG_INFO("Exited simLoop (ok=" << ok << "), destroying simulation...");
    Simulator::Destroy();
    return ok ? 0 : 1;
}
