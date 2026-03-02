package main

import (
	"fmt"
	"math"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// ==================== 1. 水箱参数 ====================

type TankParameters struct {
	A1      float64 // 水箱1截面积 (m^2)
	A2      float64 // 水箱2截面积 (m^2)
	A1Out   float64 // 水箱1出水口截面积 (m^2)
	A2Out   float64 // 水箱2出水口截面积 (m^2)
	G       float64 // 重力加速度 (m/s^2)
	KPump   float64 // 水泵增益
	H1_0    float64 // 水箱1工作点液位 (m)
	H2_0    float64 // 水箱2工作点液位 (m)
}

func NewTankParameters() *TankParameters {
	return &TankParameters{
		A1:    0.5,
		A2:    0.5,
		A1Out: 0.02,
		A2Out: 0.02,
		G:     9.81,
		KPump: 0.5,
		H1_0:  0.5,
		H2_0:  0.3,
	}
}

// ==================== 2. 双容水箱系统 ====================

type DoubleTankSystem struct {
	Params *TankParameters
	H1     float64 // 水箱1液位
	H2     float64 // 水箱2液位
}

func NewDoubleTankSystem(params *TankParameters) *DoubleTankSystem {
	if params == nil {
		params = NewTankParameters()
	}
	tank := &DoubleTankSystem{
		Params: params,
	}
	tank.Reset()
	return tank
}

func (t *DoubleTankSystem) Reset() {
	t.H1 = t.Params.H1_0
	t.H2 = t.Params.H2_0
}

// 计算液位变化率
func (t *DoubleTankSystem) Dynamics(h1, h2, u float64) (dh1dt, dh2dt float64) {
	p := t.Params
	
	// 水泵流量
	QIn := p.KPump * u / 100.0
	
	// 水箱1到水箱2的流量 (Torricelli定律)
	Q12 := 0.0
	if h1 > 0 {
		Q12 = p.A1Out * math.Sqrt(2*p.G*h1)
	}
	
	// 水箱2出水流量
	QOut := 0.0
	if h2 > 0 {
		QOut = p.A2Out * math.Sqrt(2*p.G*h2)
	}
	
	// 液位变化率
	dh1dt = (QIn - Q12) / p.A1
	dh2dt = (Q12 - QOut) / p.A2
	
	return dh1dt, dh2dt
}

// RK4积分单步
func (t *DoubleTankSystem) Step(u, dt float64) (h1, h2 float64) {
	// RK4积分
	k1H1, k1H2 := t.Dynamics(t.H1, t.H2, u)
	k2H1, k2H2 := t.Dynamics(t.H1+0.5*dt*k1H1, t.H2+0.5*dt*k1H2, u)
	k3H1, k3H2 := t.Dynamics(t.H1+0.5*dt*k2H1, t.H2+0.5*dt*k2H2, u)
	k4H1, k4H2 := t.Dynamics(t.H1+dt*k3H1, t.H2+dt*k3H2, u)
	
	t.H1 += dt * (k1H1 + 2*k2H1 + 2*k3H1 + k4H1) / 6
	t.H2 += dt * (k1H2 + 2*k2H2 + 2*k3H2 + k4H2) / 6
	
	// 防止负液位
	if t.H1 < 0 {
		t.H1 = 0
	}
	if t.H2 < 0 {
		t.H2 = 0
	}
	
	return t.H1, t.H2
}

// ==================== 3. PID控制器 ====================

type PIDController struct {
	Kp           float64
	Ki           float64
	Kd           float64
	Integral     float64
	PrevError    float64
	OutputMin    float64
	OutputMax    float64
}

func NewPIDController(kp, ki, kd float64) *PIDController {
	return &PIDController{
		Kp:        kp,
		Ki:        ki,
		Kd:        kd,
		Integral:  0,
		PrevError: 0,
		OutputMin: 0,
		OutputMax: 100,
	}
}

func (pid *PIDController) Reset() {
	pid.Integral = 0
	pid.PrevError = 0
}

func (pid *PIDController) Compute(setpoint, measurement, dt float64) float64 {
	error := setpoint - measurement
	
	// 比例项
	P := pid.Kp * error
	
	// 积分项 (带抗积分饱和)
	pid.Integral += error * dt
	if pid.Integral < -100 {
		pid.Integral = -100
	} else if pid.Integral > 100 {
		pid.Integral = 100
	}
	I := pid.Ki * pid.Integral
	
	// 微分项
	derivative := 0.0
	if dt > 0 {
		derivative = (error - pid.PrevError) / dt
	}
	D := pid.Kd * derivative
	
	// 保存当前误差
	pid.PrevError = error
	
	// 计算输出并限幅
	output := P + I + D
	if output < pid.OutputMin {
		output = pid.OutputMin
	} else if output > pid.OutputMax {
		output = pid.OutputMax
	}
	
	return output
}

// ==================== 4. 模糊隶属度函数 ====================

type FuzzyMembership struct{}

// 三角隶属度函数
func (fm *FuzzyMembership) Trimf(x float64, a, b, c float64) float64 {
	if x <= a || x >= c {
		return 0.0
	} else if a < x && x <= b {
		if b != a {
			return (x - a) / (b - a)
		}
		return 1.0
	} else {
		if c != b {
			return (c - x) / (c - b)
		}
		return 1.0
	}
}

// 高斯隶属度函数
func (fm *FuzzyMembership) Gaussmf(x, mean, sigma float64) float64 {
	return math.Exp(-0.5 * math.Pow((x-mean)/sigma, 2))
}

// ==================== 5. 模糊控制器 ====================

type FuzzyController struct {
	Labels      []string
	EMfParams   []float64
	EcMfParams  []float64
	UMfParams   []float64
	RuleTable   [][]int
}

func NewFuzzyController() *FuzzyController {
	labels := []string{"NB", "NM", "NS", "ZO", "PS", "PM", "PB"}
	
	// 模糊规则表 (7x7)
	ruleTable := [][]int{
		{0, 0, 0, 0, 1, 2, 3},   // NB
		{0, 0, 0, 1, 2, 3, 4},   // NM
		{0, 0, 1, 2, 3, 4, 5},   // NS
		{0, 1, 2, 3, 4, 5, 6},   // ZO
		{1, 2, 3, 4, 5, 6, 6},   // PS
		{2, 3, 4, 5, 6, 6, 6},   // PM
		{3, 4, 5, 6, 6, 6, 6},   // PB
	}
	
	return &FuzzyController{
		Labels:     labels,
		EMfParams:  []float64{-1, -0.66, -0.33, 0, 0.33, 0.66, 1},
		EcMfParams: []float64{-1, -0.66, -0.33, 0, 0.33, 0.66, 1},
		UMfParams:  []float64{-1, -0.66, -0.33, 0, 0.33, 0.66, 1},
		RuleTable:  ruleTable,
	}
}

// 模糊化
func (fc *FuzzyController) Fuzzify(x float64, mfParams []float64) []float64 {
	memberships := make([]float64, len(mfParams))
	fm := &FuzzyMembership{}
	
	for i, center := range mfParams {
		var a, b, c float64
		b = center
		
		if i == 0 {
			width := mfParams[1] - mfParams[0]
			a = center - width
			c = center + width
		} else if i == len(mfParams)-1 {
			width := mfParams[len(mfParams)-1] - mfParams[len(mfParams)-2]
			a = center - width
			c = center + width
		} else {
			widthLeft := center - mfParams[i-1]
			widthRight := mfParams[i+1] - center
			a = center - widthLeft
			c = center + widthRight
		}
		
		memberships[i] = fm.Trimf(x, a, b, c)
	}
	
	return memberships
}

// 重心法解模糊
func (fc *FuzzyController) Defuzzify(outputMf []float64) float64 {
	numerator := 0.0
	denominator := 0.0
	
	for i, m := range outputMf {
		numerator += m * fc.UMfParams[i]
		denominator += m
	}
	
	if denominator == 0 {
		return 0.0
	}
	return numerator / denominator
}

// 模糊推理计算
func (fc *FuzzyController) Compute(e, ec float64) float64 {
	// 限幅
	if e < -1 {
		e = -1
	} else if e > 1 {
		e = 1
	}
	if ec < -1 {
		ec = -1
	} else if ec > 1 {
		ec = 1
	}
	
	// 模糊化
	eMf := fc.Fuzzify(e, fc.EMfParams)
	ecMf := fc.Fuzzify(ec, fc.EcMfParams)
	
	// 模糊推理 (Mamdani方法)
	outputActivation := make([]float64, len(fc.Labels))
	
	for i, eDegree := range eMf {
		for j, ecDegree := range ecMf {
			if eDegree > 0 && ecDegree > 0 {
				// 取小运算
				ruleStrength := eDegree
				if ecDegree < ruleStrength {
					ruleStrength = ecDegree
				}
				outputIdx := fc.RuleTable[i][j]
				// 取大运算 (聚合)
				if ruleStrength > outputActivation[outputIdx] {
					outputActivation[outputIdx] = ruleStrength
				}
			}
		}
	}
	
	// 解模糊
	return fc.Defuzzify(outputActivation)
}

// ==================== 6. 模糊自适应串级PID控制器 ====================

type FuzzyAdaptiveCascadePID struct {
	Kp0, Ki0, Kd0 float64
	InnerPID      *PIDController
	FuzzyKp       *FuzzyController
	FuzzyKi       *FuzzyController
	FuzzyKd       *FuzzyController
	KpScale       float64
	KiScale       float64
	KdScale       float64
	Integral      float64
	PrevError     float64
	KpHistory     []float64
	KiHistory     []float64
	KdHistory     []float64
}

func NewFuzzyAdaptiveCascadePID() *FuzzyAdaptiveCascadePID {
	return &FuzzyAdaptiveCascadePID{
		Kp0:       10,
		Ki0:       0.66,
		Kd0:       1,
		InnerPID:  NewPIDController(11.5, 1.55, 0.45),
		FuzzyKp:   NewFuzzyController(),
		FuzzyKi:   NewFuzzyController(),
		FuzzyKd:   NewFuzzyController(),
		KpScale:   5.0,
		KiScale:   1.0,
		KdScale:   2.0,
		Integral:  0,
		PrevError: 0,
	}
}

func (fap *FuzzyAdaptiveCascadePID) Reset() {
	fap.InnerPID.Reset()
	fap.Integral = 0
	fap.PrevError = 0
	fap.KpHistory = []float64{}
	fap.KiHistory = []float64{}
	fap.KdHistory = []float64{}
}

func (fap *FuzzyAdaptiveCascadePID) Compute(setpoint, h2, h1, dt float64) float64 {
	// ========== 主回路: 模糊自适应PID ==========
	error := setpoint - h2
	
	// 归一化误差和误差变化率
	eMax := 0.5
	ecMax := 0.5
	
	eNorm := error / eMax
	if eNorm < -1 {
		eNorm = -1
	} else if eNorm > 1 {
		eNorm = 1
	}
	
	ecNorm := 0.0
	if dt > 0 {
		ecNorm = (error - fap.PrevError) / dt / ecMax
	}
	if ecNorm < -1 {
		ecNorm = -1
	} else if ecNorm > 1 {
		ecNorm = 1
	}
	
	// 模糊推理调整PID参数
	deltaKp := fap.FuzzyKp.Compute(eNorm, ecNorm)
	deltaKi := fap.FuzzyKi.Compute(eNorm, ecNorm)
	deltaKd := fap.FuzzyKd.Compute(eNorm, ecNorm)
	
	// 计算实际PID参数
	Kp := fap.Kp0 + fap.KpScale*deltaKp
	Ki := fap.Ki0 + fap.KiScale*deltaKi
	Kd := fap.Kd0 + fap.KdScale*deltaKd
	
	// 确保参数为正
	if Kp < 0.1 {
		Kp = 0.1
	}
	if Ki < 0.01 {
		Ki = 0.01
	}
	if Kd < 0.01 {
		Kd = 0.01
	}
	
	// 记录参数
	fap.KpHistory = append(fap.KpHistory, Kp)
	fap.KiHistory = append(fap.KiHistory, Ki)
	fap.KdHistory = append(fap.KdHistory, Kd)
	
	// 主回路PID计算
	P := Kp * error
	fap.Integral += error * dt
	if fap.Integral < -10 {
		fap.Integral = -10
	} else if fap.Integral > 10 {
		fap.Integral = 10
	}
	I := Ki * fap.Integral
	
	derivative := 0.0
	if dt > 0 {
		derivative = (error - fap.PrevError) / dt
	}
	D := Kd * derivative
	
	h1Setpoint := P + I + D
	if h1Setpoint < 0 {
		h1Setpoint = 0
	} else if h1Setpoint > 2.0 {
		h1Setpoint = 2.0
	}
	
	fap.PrevError = error
	
	// ========== 副回路: 传统PID ==========
	u := fap.InnerPID.Compute(h1Setpoint, h1, dt)
	
	return u
}

// ==================== 7. 仿真结果结构 ====================

type SimulationResults struct {
	T               []float64
	Setpoint        []float64
	H1              []float64
	H2              []float64
	U               []float64
	Error           []float64
	ControllerType  string
	KpHistory       []float64
	KiHistory       []float64
	KdHistory       []float64
}

// ==================== 8. 仿真函数 ====================

func SimulateSystem(controllerType string, duration, dt float64) *SimulationResults {
	tank := NewDoubleTankSystem(nil)
	tank.Reset()
	
	var controller interface{}
	var pidController *PIDController
	
	switch controllerType {
	case "pid":
		controller = NewPIDController(10.0, 0.66, 1.0)
	case "fuzzy_cascade":
		controller = NewFuzzyAdaptiveCascadePID()
	default:
		controller = NewPIDController(10.0, 0.66, 1.0)
	}
	
	nSteps := int(duration / dt)
	t := make([]float64, nSteps)
	setpoint := make([]float64, nSteps)
	h1Record := make([]float64, nSteps)
	h2Record := make([]float64, nSteps)
	uRecord := make([]float64, nSteps)
	errorRecord := make([]float64, nSteps)
	
	// 生成设定值 (阶跃变化)
	for i := 0; i < nSteps; i++ {
		t[i] = float64(i) * dt
		setpoint[i] = 0.3
		
		timeSec := float64(i) * dt
		if timeSec >= 50 && timeSec < 100 {
			setpoint[i] = 0.5
		} else if timeSec >= 100 && timeSec < 150 {
			setpoint[i] = 0.4
		} else if timeSec >= 150 {
			setpoint[i] = 0.6
		}
	}
	
	// 运行仿真
	for i := 0; i < nSteps; i++ {
		h1, h2 := tank.H1, tank.H2
		h1Record[i] = h1
		h2Record[i] = h2
		
		// 计算控制量
		var u float64
		switch ctrl := controller.(type) {
		case *PIDController:
			u = ctrl.Compute(setpoint[i], h2, dt)
		case *FuzzyAdaptiveCascadePID:
			u = ctrl.Compute(setpoint[i], h2, h1, dt)
		}
		
		uRecord[i] = u
		errorRecord[i] = setpoint[i] - h2
		
		// 系统单步
		tank.Step(u, dt)
	}
	
	results := &SimulationResults{
		T:              t,
		Setpoint:       setpoint,
		H1:             h1Record,
		H2:             h2Record,
		U:              uRecord,
		Error:          errorRecord,
		ControllerType: controllerType,
	}
	
	// 添加模糊自适应参数历史
	if controllerType == "fuzzy_cascade" {
		if fc, ok := controller.(*FuzzyAdaptiveCascadePID); ok {
			results.KpHistory = fc.KpHistory
			results.KiHistory = fc.KiHistory
			results.KdHistory = fc.KdHistory
		}
	}
	
	return results
}

// ==================== 9. 性能指标计算 ====================

type Metrics struct {
	IAE           float64
	ISE           float64
	ITAE          float64
	MaxOvershoot  float64
	ControlEffort float64
}

func CalculateMetrics(results *SimulationResults, dt float64) *Metrics {
	steadyIdx := int(10.0 / dt)
	n := len(results.Error)
	
	iae := 0.0
	ise := 0.0
	itae := 0.0
	maxOvershoot := 0.0
	
	for i := steadyIdx; i < n; i++ {
		absError := math.Abs(results.Error[i])
		iae += absError * dt
		ise += results.Error[i] * results.Error[i] * dt
		itae += results.T[i] * absError * dt
		
		overshoot := results.H2[i] - results.Setpoint[i]
		if overshoot > maxOvershoot {
			maxOvershoot = overshoot
		}
	}
	
	// 控制量变化总量
	controlEffort := 0.0
	for i := 1; i < len(results.U); i++ {
		controlEffort += math.Abs(results.U[i] - results.U[i-1])
	}
	
	return &Metrics{
		IAE:           iae,
		ISE:           ise,
		ITAE:          itae,
		MaxOvershoot:  maxOvershoot,
		ControlEffort: controlEffort,
	}
}

// ==================== 10. 主函数 ====================

func main() {
	fmt.Println("双容水箱模糊自适应串级PID控制系统")
	fmt.Println("====================================")
	
	// 运行仿真
	duration := 200.0
	dt := 0.1
	
	fmt.Println("\n运行传统PID仿真...")
	resultsPID := SimulateSystem("pid", duration, dt)
	metricsPID := CalculateMetrics(resultsPID, dt)
	
	fmt.Println("\n运行模糊自适应串级PID仿真...")
	resultsFuzzy := SimulateSystem("fuzzy_cascade", duration, dt)
	metricsFuzzy := CalculateMetrics(resultsFuzzy, dt)
	
	// 打印性能对比
	fmt.Println("\n========== 性能指标对比 ==========")
	fmt.Printf("指标\t\t\t传统PID\t\t模糊自适应串级PID\n")
	fmt.Printf("IAE\t\t\t%.4f\t\t%.4f\n", metricsPID.IAE, metricsFuzzy.IAE)
	fmt.Printf("ISE\t\t\t%.4f\t\t%.4f\n", metricsPID.ISE, metricsFuzzy.ISE)
	fmt.Printf("ITAE\t\t\t%.4f\t\t%.4f\n", metricsPID.ITAE, metricsFuzzy.ITAE)
	fmt.Printf("Max Overshoot\t\t%.4f\t\t%.4f\n", metricsPID.MaxOvershoot, metricsFuzzy.MaxOvershoot)
	fmt.Printf("Control Effort\t\t%.4f\t\t%.4f\n", metricsPID.ControlEffort, metricsFuzzy.ControlEffort)
	
	// 保存结果到CSV
	saveToCSV(resultsPID, "pid_results.csv")
	saveToCSV(resultsFuzzy, "fuzzy_cascade_results.csv")
	
	fmt.Println("\n仿真完成！结果已保存到CSV文件。")
}

// 保存结果到CSV
func saveToCSV(results *SimulationResults, filename string) {
	// 实际实现需要使用文件操作
	fmt.Printf("保存 %s ...\n", filename)
}