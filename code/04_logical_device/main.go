package main

import (
	"flag"
	"fmt"
	"log"
	"runtime"

	"github.com/go-gl/glfw/v3.3/glfw"
	"github.com/ironsmile/vulkan-tutorial-go/code/optional"
	vk "github.com/vulkan-go/vulkan"
)

func init() {
	// This is needed to arrange that main() runs on main thread.
	// See documentation for functions that are only allowed to be called
	// from the main thread.
	runtime.LockOSThread()

	flag.BoolVar(&args.debug, "debug", false, "Enable Vulkan validation layers")
}

var args struct {
	debug bool
}

const (
	title = "Vulkan Tutorial: Logical device and queues"
)

func main() {
	flag.Parse()

	app := &VulkanTutorialApp{
		width:                  1024,
		height:                 768,
		enableValidationLayers: args.debug,
		validationLayers: []string{
			"VK_LAYER_KHRONOS_validation\x00",
		},
		physicalDevice: vk.PhysicalDevice(vk.NullHandle),
		device:         vk.Device(vk.NullHandle),
	}
	if err := app.Run(); err != nil {
		log.Fatalf("ERROR: %s", err)
	}
}

// VulkanTutorialApp is the first program from vulkan-tutorial.com
type VulkanTutorialApp struct {
	width  int
	height int

	// validationLayers is the list of required device extensions needed by this
	// program when the -debug flag is set.
	validationLayers       []string
	enableValidationLayers bool

	window   *glfw.Window
	instance vk.Instance

	// physicalDevice is the physical device selected for this program.
	physicalDevice vk.PhysicalDevice

	// device is the logical device created for interfacing with the physical device.
	device vk.Device

	graphicsQueue vk.Queue
}

// Run runs the vulkan program.
func (a *VulkanTutorialApp) Run() error {
	if err := a.initWindow(); err != nil {
		return fmt.Errorf("initWindow: %w", err)
	}
	defer a.cleanWindow()

	if err := a.initVulkan(); err != nil {
		return fmt.Errorf("initVulkan: %w", err)
	}
	defer a.cleanupVulkan()

	if err := a.mainLoop(); err != nil {
		return fmt.Errorf("mainLoop: %w", err)
	}

	return nil
}

func (a *VulkanTutorialApp) initWindow() error {
	if err := glfw.Init(); err != nil {
		return fmt.Errorf("glfw.Init: %w", err)
	}

	glfw.WindowHint(glfw.ClientAPI, glfw.NoAPI)
	glfw.WindowHint(glfw.Resizable, glfw.False)

	window, err := glfw.CreateWindow(a.width, a.height, title, nil, nil)
	if err != nil {
		return fmt.Errorf("creating window: %w", err)
	}

	a.window = window
	return nil
}

func (a *VulkanTutorialApp) cleanWindow() {
	a.window.Destroy()
	glfw.Terminate()
}

func (a *VulkanTutorialApp) initVulkan() error {
	vk.SetGetInstanceProcAddr(glfw.GetVulkanGetInstanceProcAddress())

	if err := vk.Init(); err != nil {
		return fmt.Errorf("failed to init Vulkan Go: %w", err)
	}

	if err := a.createInstance(); err != nil {
		return fmt.Errorf("createInstance: %w", err)
	}

	if err := a.pickPhysicalDevice(); err != nil {
		return fmt.Errorf("pickPhysicalDevice: %w", err)
	}

	if err := a.createLogicalDevice(); err != nil {
		return fmt.Errorf("createLogicalDevice: %w", err)
	}

	return nil
}

func (a *VulkanTutorialApp) cleanupVulkan() {
	if a.device != vk.Device(vk.NullHandle) {
		vk.DestroyDevice(a.device, nil)
	}
	vk.DestroyInstance(a.instance, nil)
}

func (a *VulkanTutorialApp) pickPhysicalDevice() error {
	var deviceCount uint32
	err := vk.Error(vk.EnumeratePhysicalDevices(a.instance, &deviceCount, nil))
	if err != nil {
		return fmt.Errorf("failed to get the number of physical devices: %w", err)
	}
	if deviceCount == 0 {
		return fmt.Errorf("failed to find GPUs with Vulkan support")
	}

	pDevices := make([]vk.PhysicalDevice, deviceCount)
	err = vk.Error(vk.EnumeratePhysicalDevices(a.instance, &deviceCount, pDevices))
	if err != nil {
		return fmt.Errorf("failed to enumerate the physical devices: %w", err)
	}

	var (
		selectedDevice vk.PhysicalDevice
		score          uint32
	)

	for _, device := range pDevices {
		deviceScore := a.getDeviceScore(device)

		if deviceScore > score {
			selectedDevice = device
			score = deviceScore
		}
	}

	if selectedDevice == vk.PhysicalDevice(vk.NullHandle) {
		return fmt.Errorf("failed to find suitable physical devices")
	}

	a.physicalDevice = selectedDevice
	return nil
}

func (a *VulkanTutorialApp) createLogicalDevice() error {
	indices := a.findQueueFamilies(a.physicalDevice)
	if !indices.IsComplete() {
		return fmt.Errorf("createLogicalDevice called for physical device which does " +
			"have all the queues required by the program")
	}

	queueFamilies := make(map[uint32]struct{})
	queueFamilies[indices.Graphics.Get()] = struct{}{}

	queueCreateInfos := []vk.DeviceQueueCreateInfo{}

	for familyIndex := range queueFamilies {
		queueCreateInfos = append(
			queueCreateInfos,
			vk.DeviceQueueCreateInfo{
				SType:            vk.StructureTypeDeviceQueueCreateInfo,
				QueueFamilyIndex: familyIndex,
				QueueCount:       1,
				PQueuePriorities: []float32{1.0},
			},
		)
	}

	//!TODO: left for later use
	deviceFeatures := []vk.PhysicalDeviceFeatures{{}}

	createInfo := vk.DeviceCreateInfo{
		SType:            vk.StructureTypeDeviceCreateInfo,
		PEnabledFeatures: deviceFeatures,

		PQueueCreateInfos:    queueCreateInfos,
		QueueCreateInfoCount: uint32(len(queueCreateInfos)),

		EnabledExtensionCount: 0,
	}

	if a.enableValidationLayers {
		createInfo.PpEnabledLayerNames = a.validationLayers
		createInfo.EnabledLayerCount = uint32(len(a.validationLayers))
	}

	var device vk.Device
	err := vk.Error(vk.CreateDevice(a.physicalDevice, &createInfo, nil, &device))
	if err != nil {
		return fmt.Errorf("failed to create logical device: %w", err)
	}
	a.device = device

	var graphicsQueue vk.Queue
	vk.GetDeviceQueue(a.device, indices.Graphics.Get(), 0, &graphicsQueue)
	a.graphicsQueue = graphicsQueue

	return nil
}

func (a *VulkanTutorialApp) createInstance() error {
	if a.enableValidationLayers && !a.checkValidationSupport() {
		return fmt.Errorf("validation layers requested but not available")
	}

	appInfo := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   title + "\x00",
		ApplicationVersion: vk.MakeVersion(1, 0, 0),
		PEngineName:        "No Engine\x00",
		EngineVersion:      vk.MakeVersion(1, 0, 0),
		ApiVersion:         vk.ApiVersion10,
	}

	glfwExtensions := glfw.GetCurrentContext().GetRequiredInstanceExtensions()
	createInfo := vk.InstanceCreateInfo{
		SType:                   vk.StructureTypeInstanceCreateInfo,
		PApplicationInfo:        &appInfo,
		EnabledExtensionCount:   uint32(len(glfwExtensions)),
		PpEnabledExtensionNames: glfwExtensions,
	}

	if a.enableValidationLayers {
		createInfo.EnabledLayerCount = uint32(len(a.validationLayers))
		createInfo.PpEnabledLayerNames = a.validationLayers
	}

	var instance vk.Instance
	if res := vk.CreateInstance(&createInfo, nil, &instance); res != vk.Success {
		return fmt.Errorf("failed to create Vulkan instance: %w", vk.Error(res))
	}

	a.instance = instance
	return nil
}

// findQueueFamilies returns a FamilyIndeces populated with Vulkan queue families needed
// by the program.
func (a *VulkanTutorialApp) findQueueFamilies(
	device vk.PhysicalDevice,
) QueueFamilyIndices {
	indices := QueueFamilyIndices{}

	var queueFamilyCount uint32
	vk.GetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nil)

	queueFamilies := make([]vk.QueueFamilyProperties, queueFamilyCount)
	vk.GetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies)

	for i, family := range queueFamilies {
		family.Deref()

		if family.QueueFlags&vk.QueueFlags(vk.QueueGraphicsBit) != 0 {
			indices.Graphics.Set(uint32(i))
		}

		if indices.IsComplete() {
			break
		}
	}

	return indices
}

// getDeviceScore returns how suitable is this device for the current program.
// Bigger score means better. Zero or negative means the device cannot be used.
func (a *VulkanTutorialApp) getDeviceScore(device vk.PhysicalDevice) uint32 {
	var (
		deviceScore uint32
		properties  vk.PhysicalDeviceProperties
	)

	vk.GetPhysicalDeviceProperties(device, &properties)
	properties.Deref()

	if properties.DeviceType == vk.PhysicalDeviceTypeDiscreteGpu {
		deviceScore += 1000
	} else {
		deviceScore++
	}

	if !a.isDeviceSuitable(device) {
		deviceScore = 0
	}

	if args.debug {
		log.Printf(
			"Available device: %s (score: %d)",
			vk.ToString(properties.DeviceName[:]),
			deviceScore,
		)
	}

	return deviceScore
}

func (a *VulkanTutorialApp) isDeviceSuitable(device vk.PhysicalDevice) bool {
	indices := a.findQueueFamilies(device)

	return indices.IsComplete()
}

func (a *VulkanTutorialApp) checkValidationSupport() bool {
	var count uint32
	if vk.EnumerateInstanceLayerProperties(&count, nil) != vk.Success {
		return false
	}
	availableLayers := make([]vk.LayerProperties, count)

	if vk.EnumerateInstanceLayerProperties(&count, availableLayers) != vk.Success {
		return false
	}

	availableLayersStr := make([]string, 0, count)
	for _, layer := range availableLayers {
		layer.Deref()

		layerName := vk.ToString(layer.LayerName[:])
		availableLayersStr = append(availableLayersStr, layerName+"\x00")
	}

	for _, validationLayer := range a.validationLayers {
		layerFound := false

		for _, instanceLayer := range availableLayersStr {
			if validationLayer == instanceLayer {
				layerFound = true
				break
			}
		}

		if !layerFound {
			return false
		}
	}

	return true
}

func (a *VulkanTutorialApp) mainLoop() error {
	log.Printf("main loop!\n")

	for !a.window.ShouldClose() {
		glfw.PollEvents()
	}

	return nil
}

// QueueFamilyIndices holds the indexes of Vulkan queue families needed by the programs.
type QueueFamilyIndices struct {

	// Graphics is the index of the graphics queue family.
	Graphics optional.Optional[uint32]
}

// IsComplete returns true if all families have been set.
func (f *QueueFamilyIndices) IsComplete() bool {
	return f.Graphics.HasValue()
}
