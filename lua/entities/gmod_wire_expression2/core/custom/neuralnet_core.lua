-- Made by KrypteK
E2Lib.RegisterExtension("neuralnet", false, "Adds neural networks to E2")

registerType("neuralnet", "xnn", {},
    nil,
    nil,
    function(retval)
        if not (type(retval) == "xnn") then error("Return value is not a neuralnet, but a "..type(retval).."!",0) end
    end,
    function(v)
        return not (type(retval) == "xnn")
    end
)

registerOperator("ass", "xnn", "xnn", function(self, args)
	local op1, op2, scope = args[2], args[3], args[4]
	local      rv2 = op2[1](self, op2)
	self.Scopes[scope][op1] = rv2
    self.Scopes[scope].vclk[op1] = true

	return rv2
end)

--local cvar_equidistant = CreateConVar( "wire_expression2_curves_mindistance", "5", FCVAR_ARCHIVE )
--local cvar_maxsteps = CreateConVar( "wire_expression2_curves_maxsteps", "100", FCVAR_ARCHIVE )

local cvar_maxiterations = CreateConVar( "wire_expression2_neuralnet_maxiterations", "25", FCVAR_ARCHIVE )

--------------------------------------------------------------------------------

Matrix = {}
Matrix.__index = Matrix

function Matrix:create (rows, cols) 
    local m = {}
    m.rows = rows
    m.cols = cols

    m.data = {}

    for i = 1,rows do
        m.data[i] = {}

        for j = 1,cols do
            m.data[i][j] = 0
        end
    end

    return setmetatable(m, Matrix)
end

function Matrix:Randomize () 
    for i = 1,self.rows do
        for j = 1,self.cols do
            self.data[i][j] = math.Rand( -1, 1 )
        end
    end
end

function Matrix:FromArray (arr)
    local m = Matrix:create(#arr,1)
       
    for i = 1,#arr do
        m.data[i][1] = arr[i]
    end

    return m
end

function Matrix:toArray ()
    local arr = {}
       
    for i = 1,self.rows do
        for j = 1,self.cols do         
            table.insert(arr, self.data[i][j])
        end
    end

    return arr
end

function Matrix:subtractMatrix (m) 
    if self.rows ~= m.rows and self.cols ~= m.cols then
        error("Columns and Rows of A must match Columns and Rows of B.")
    end

    for i = 1,self.rows do
        for j = 1,self.cols do
            self.data[i][j] = self.data[i][j] - m.data[i][j]
        end
    end
end

function Matrix:SubtractMatrices (a, b) 
    if a.rows ~= b.rows and a.cols ~= b.cols then
        error("Columns and Rows of A must match Columns and Rows of B.")
    end

    local new = Matrix:create(a.rows,a.cols)

    for i = 1, a.rows do
        for j = 1,a.cols do
            new.data[i][j] = a.data[i][j] - b.data[i][j]
        end
    end

    return new
end

function Matrix:addMatrix (m) 
    if self.rows ~= m.rows and self.cols ~= m.cols then
        error("Columns and Rows of A must match Columns and Rows of B.")
    end

    for i = 1,self.rows do
        for j = 1,self.cols do
            self.data[i][j] = self.data[i][j] + m.data[i][j]
        end
    end
end

function Matrix:MultiplyMatrices (a, b) 
    if a.cols ~= b.rows then
        error("Columns of A must match rows of B.")
    end  
    
    local new = Matrix:create(a.rows, b.cols)
       
    for i = 1, a.rows do 
        for j = 1,b.cols do                             
            for k = 1,a.cols do
                new.data[i][j] = 
                new.data[i][j] + a.data[i][k] * b.data[k][j]
            end
        end          
    end
        
    return new   
end

function Matrix:multiplyMatrix (m) 
    if self.rows ~= m.rows and self.cols ~= m.cols then
        error("Columns and Rows of A must match Columns and Rows of B.")
    end

    for i = 1,self.rows do
        for j = 1,self.cols do
            self.data[i][j] = self.data[i][j] * m.data[i][j]
        end
    end
end

function Matrix:multiplyNumber (n) 
    for i = 1,self.rows do
        for j = 1,self.cols do
            self.data[i][j] = self.data[i][j] * n
        end
    end
end

function Matrix:Transpose (m) 
    local new = Matrix:create(m.cols, m.rows)

    for i = 1,new.rows do
        for j = 1,new.cols do
            new.data[i][j] = m.data[j][i]
        end
    end

    return new
end

function Matrix:map (func) 
    for i = 1,self.rows do
        for j = 1,self.cols do
            self.data[i][j] = func(self.data[i][j])
        end
    end
end

function Matrix:MapMatrix (m, func) 
    local new = Matrix:create(m.rows, m.cols)

    for i = 1,new.rows do
        for j = 1,new.cols do
            new.data[i][j] = func(m.data[i][j])
        end
    end

    return new
end

local DEFAULT = {n={},ntypes={},s={},stypes={},size=0}

function Matrix:toTable()  
    local ret = table.Copy(DEFAULT)

    ret.s["cols"] = self.cols
    ret.stypes["cols"] = "n"  
    ret.s["rows"] = self.rows
    ret.stypes["rows"] = "n"  
    ret.s["data"] = table.Copy(DEFAULT)
    ret.stypes["data"] = "t"  

    local data = ret.s["data"]

    for k,v in pairs(self.data) do
        data.n[k] = table.Copy(DEFAULT)
        data.ntypes[k] = "t"   

        local datarow = data.n[k]
        local row = self.data[k]
    
        for i,j in pairs(row) do
            datarow.n[i] = j
            datarow.ntypes[i] = "n"         
        end  
    end   

    return ret
end

function Matrix:FromTable(t)
    local ret = Matrix:create(t.s["rows"],t.s["cols"])

    for i = 1,ret.rows do
        for j = 1,ret.cols do
            ret.data[i][j] = t.s["data"].n[i].n[j]
        end
    end

    return ret
end

function ternary ( cond , T , F )
    if cond then return T else return F end
end

ActivationFunctions = {}

ActivationFunctions.Sigmoid = {
    Name = "Sigmoid",
    Equation = function (x) 
        return 1 / (1 + math.exp(-x))
    end,
    Derivative = function (x) 
        return x * (1 - x)
    end,
}

ActivationFunctions.Softsign = {
    Name = "Softsign",
    Equation = function (x) 
        return 1 / (1 + math.abs(x))
    end,
    Derivative = function (x) 
        return math.pow(1 / (1 + math.abs(x)),2)
    end,
}

ActivationFunctions.ArcTan = {
    Name = "ArcTan",
    Equation = function (x) 
        return math.atan(x)
    end,
    Derivative = function (x) 
        return 1 / (math.pow(x,2) + 1)
    end,
}

ActivationFunctions.Tanh = {
    Name = "Tanh",
    Equation = function (x) 
        return math.tanh(x)
    end,
    Derivative = function (x) 
        return 1 - (x * x)
    end,
}

ActivationFunctions.ReLU = {
    Name = "ReLU",
    Equation = function (x) 
        return ternary(x < 0, 0, x)
    end,
    Derivative = function (x) 
        return ternary(x < 0, 0, 1)
    end,
}

ActivationFunctions.ISRU = {
    Name = "ISRU",
    Equation = function (x) 
        return x / math.sqrt(1 + math.pow(x,2))
    end,
    Derivative = function (x) 
        return math.pow(x / math.sqrt(1 + math.pow(x,2)),3)
    end,
}

ActivationFunctions.Sin = {
    Name = "Sin",
    Equation = function (x) 
        return math.sin(x)
    end,
    Derivative = function (x) 
        return math.cos(x)
    end,
}

ActivationFunctions.Guassian = {
    Name = "Guassian",
    Equation = function (x) 
        return math.exp(-math.pow(x,2))
    end,
    Derivative = function (x) 
        return -2 * x * math.exp(-math.pow(x,2))
    end,
}

function GetActivationFunction (name) 
    for k,v in pairs(ActivationFunctions) do
        if v.Name:lower() == name:lower() then
            return v
        end
    end

    error("Invalid activation function.")
end

NeuralNetwork = {}
NeuralNetwork.__index = NeuralNetwork

function NeuralNetwork:create (args) 
    if #args>=3 then
        local nn = {}
        setmetatable(nn, NeuralNetwork)

        nn.Structure = args
        nn.Weights = {}
        nn.Bias = {}
        nn.Iteration = 0
        nn.LearningRate = 0.03
        nn.ActivationFunction = ActivationFunctions.Sigmoid

        for i = 1,#args - 1 do
            if not (type(args[i]) == "number") then error("Expected number for layer.") end
            
            local weights = Matrix:create(nn.Structure[i + 1], nn.Structure[i])
            weights:Randomize()

            local bias = Matrix:create(nn.Structure[i + 1], 1)
            bias:Randomize()

            nn.Weights[i] = weights
            nn.Bias[i] = bias
        end

        return nn
    else
        error("Expected atleast 3 layers.")
    end
end

function NeuralNetwork:predict (args)
    if #args ~= self.Structure[1] then
        error("Input count must match node count of first layer.")
    end

    for k,v in pairs(args) do
        if not (type(v) == "number") then error("Expected number.") end
    end

    local matrix = Matrix:FromArray(args)      

    for i = 1,#self.Structure - 1 do     
        local matrixNext = Matrix:MultiplyMatrices(self.Weights[i],matrix)             
        
        matrixNext:addMatrix(self.Bias[i])
        matrixNext:map(self.ActivationFunction.Equation)

        matrix = matrixNext
    end  

    return matrix:toArray()   
end

function NeuralNetwork:train (args, target)
    if #args ~= self.Structure[1] then
        error("Input count must match node count of the first layer.")
    end

    if #target ~= self.Structure[#self.Structure] then
        error("Target count must match node count of the last layer.")
    end

    for k,v in pairs(args) do if not (type(v) == "number") then error("Input and target arrays may only contain numbers.") end end
    for k,v in pairs(target) do if not (type(v) == "number") then error("Input and target arrays may only contain numbers.") end end
    
    local layers = {}
    layers[1] = Matrix:FromArray(args)

    for i = 2, #self.Structure do
        layers[i] = Matrix:MultiplyMatrices(self.Weights[i - 1], layers[i - 1])
        layers[i]:addMatrix(self.Bias[i-1])
        layers[i]:map(self.ActivationFunction.Equation)
    end

    local error = Matrix:SubtractMatrices(Matrix:FromArray(target), layers[#layers])
    local i = #self.Structure
  
    while i >= 2 do
        local gradients = Matrix:MapMatrix(layers[i], self.ActivationFunction.Derivative) 
        gradients:multiplyMatrix(error)
        gradients:multiplyNumber(self.LearningRate)

        local hiddenTranspose = Matrix:Transpose(layers[i - 1])
        local deltas = Matrix:MultiplyMatrices(gradients, hiddenTranspose)

        self.Weights[i - 1]:addMatrix(deltas)
        self.Bias[i - 1]:addMatrix(gradients)

        if i > 2 then
            error = Matrix:MultiplyMatrices(Matrix:Transpose(self.Weights[i-1]), error)
        end

        i = i - 1
    end

    self.Iteration = self.Iteration + 1
end

--------------------------------------------------------------------------------
__e2setcost(20)
e2function neuralnet createNeuralNetwork(...)
    return NeuralNetwork:create({...})
end

__e2setcost(20)
e2function neuralnet createNeuralNetwork(neuralnet nn)
    local ret = NeuralNetwork:create(nn.Structure)

    ret.LearningRate = nn.LearningRate
    ret.ActivationFunction = nn.ActivationFunction
    ret.Iteration = nn.Iteration

    for k,v in pairs(ret.Weights) do
        for i = 1,v.rows do
            for j = 1,v.cols do
                v.data[i][j] = nn.Weights[k].data[i][j]
            end
        end 
    end  

    for k,v in pairs(ret.Bias) do
        for i = 1,v.rows do
            for j = 1,v.cols do
                v.data[i][j] = nn.Bias[k].data[i][j]
            end
        end 
    end  

    return ret
end

__e2setcost(20)
e2function table neuralnet:toTable()
    if not this then return {} end

    local ret = table.Copy(DEFAULT)

    ret.s["Iteration"] = this.Iteration
    ret.stypes["Iteration"] = "n"

    ret.s["LearningRate"] = this.LearningRate
    ret.stypes["LearningRate"] = "n"

    ret.s["Activation"] = this.ActivationFunction.Name
    ret.stypes["Activation"] = "s"

    ret.s["Structure"] = table.Copy(DEFAULT)
    ret.stypes["Structure"] = "t"

    local structure = ret.s["Structure"] 

    for k,v in pairs(this.Structure) do
        structure.n[k] = v
        structure.ntypes[k] = "n"      
    end   

    ret.s["Weights"] = table.Copy(DEFAULT)
    ret.stypes["Weights"] = "t"

    local weights = ret.s["Weights"] 

    for k,v in pairs(this.Weights) do
        weights.n[k] = v:toTable() 
        weights.ntypes[k] = "t"      
    end   

    ret.s["Bias"] = table.Copy(DEFAULT)
    ret.stypes["Bias"] = "t"

    local bias = ret.s["Bias"] 

    for k,v in pairs(this.Bias) do
        bias.n[k] = v:toTable() 
        bias.ntypes[k] = "t"      
    end  

    ret.size = #this

    PrintTable(ret)

    return ret
end

__e2setcost(20)
e2function neuralnet createNeuralNetwork(table t)
    local ret = NeuralNetwork:create(t.s["Structure"].n)

    ret.ActivationFunction = GetActivationFunction(t.s["Activation"])
    ret.LearningRate = t.s["LearningRate"]
    ret.Iteration = t.s["Iteration"]

    for k,v in pairs(ret.Weights) do
        for i = 1,v.rows do
            for j = 1,v.cols do
                v.data[i][j] = t.s["Weights"].n[k].s["data"].n[i].n[j]
            end
        end 
    end  

    for k,v in pairs(ret.Bias) do
        for i = 1,v.rows do
            for j = 1,v.cols do
                v.data[i][j] = t.s["Bias"].n[k].s["data"].n[i].n[j]
            end
        end 
    end  

    print(ret.Bias)

    return ret
end

__e2setcost(1)
e2function void neuralnet:setLearningRate(number n) 
    if not this then return end

    this.LearningRate = n
end

__e2setcost(1)
e2function void neuralnet:setActivationFunction(string name) 
    if not this then return end

    this.ActivationFunction = GetActivationFunction(name)
end

__e2setcost(25)
e2function array neuralnet:predict(array input) 
    if not this then return {} end

    return this:predict(input)
end

e2function array neuralnet:predict(...) 
    if not this then return {} end

    return this:predict({...})
end

__e2setcost(100)
e2function void neuralnet:train(array input, array target) 
    if not this then return end

    this:train(input, target)
end

e2function void neuralnet:train(array input, array target, number iterations) 
    if not this then return end

    local max =  math.min(iterations, cvar_maxiterations:GetInt())

    for i = 1, max do
        self.prf = self.prf + 20 
        this:train(input, target)
    end
end

__e2setcost(1)
e2function number neuralnet:iteration() 
    if not this then return 0 end

    return this.Iteration
end

e2function number neuralnet:learningRate() 
    if not this then return 0 end

    return this.LearningRate
end

e2function string neuralnet:activationFunction() 
    if not this then return 0 end

    return this.ActivationFunction.Name
end

e2function array neuralnet:structure() 
    if not this then return 0 end

    return this.structure
end

e2function void neuralnet:print() 
    PrintTable(this)
end