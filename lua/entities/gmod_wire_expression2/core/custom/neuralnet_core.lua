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

    for i = 1,m.rows do
        for j = 1,m.cols do
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

Sigmoid = {
    Name = "Sigmoid",
    Equation = function (x) 
        return 1 / (1 + math.exp(-x))
    end,
    Derivative = function (x) 
        return x * (1 - x)
    end,
}

--------------------------------------------------------------------------------
__e2setcost(2)

e2function neuralnet createNeuralNetwork(...)
    local args = {...}

    if #args>=3 then
        local nn = {}

        nn.Structure = args
        nn.Weights = {}
        nn.Bias = {}

        for i = 1,#args - 1 do
            local v = args[i]

            if not (type(v) == "number") then error("Expected number for layer.") end
            
            local weights = Matrix:create(nn.Structure[i + 1], nn.Structure[i])
            weights:Randomize()

            nn.Weights[i] = weights

            local bias = Matrix:create(nn.Structure[i + 1], 1)
            bias:Randomize()

            nn.Bias[i] = bias
        end

        nn.LearningRate = 0.1
        nn.ActivationFunction = Sigmoid

        return nn
    else
        error("Expected atleast 3 layers.")
    end
end

e2function array neuralnet:predict(array input) 
    local matrix = Matrix:FromArray(input)      

    for i = 1,#this.Structure - 1 do     
        local matrixNext = Matrix:MultiplyMatrices(this.Weights[i],matrix)             
        
        matrixNext:addMatrix(this.Bias[i])
        matrixNext:map(this.ActivationFunction.Equation)

        matrix = matrixNext
    end  

    return matrix:toArray()
end

-- e2function string toString(neuralnet nn)
--     return "Gay"
-- end

-- --- Gets the vector nicely formatted as a string "[X,Y,Z]"
-- e2function string neuralnet:toString() = e2function string toString(neuralnet v)
