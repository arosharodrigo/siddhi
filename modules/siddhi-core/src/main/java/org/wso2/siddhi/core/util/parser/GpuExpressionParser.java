package org.wso2.siddhi.core.util.parser;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.apache.log4j.Logger;
import org.wso2.siddhi.core.config.SiddhiContext;
import org.wso2.siddhi.core.event.ComplexMetaEvent;
import org.wso2.siddhi.core.event.stream.MetaStreamEvent;
import org.wso2.siddhi.core.exception.OperationNotSupportedException;
import org.wso2.siddhi.core.exception.QueryCreationException;
import org.wso2.siddhi.core.executor.ConstantExpressionExecutor;
import org.wso2.siddhi.core.executor.ExpressionExecutor;
import org.wso2.siddhi.core.executor.VariableExpressionExecutor;
import org.wso2.siddhi.core.executor.condition.AndConditionExpressionExecutor;
import org.wso2.siddhi.core.executor.condition.ConditionExpressionExecutor;
import org.wso2.siddhi.core.executor.condition.NotConditionExpressionExecutor;
import org.wso2.siddhi.core.executor.condition.OrConditionExpressionExecutor;
import org.wso2.siddhi.core.executor.condition.compare.contains.ContainsCompareConditionExpressionExecutor;
import org.wso2.siddhi.core.executor.condition.compare.equal.EqualCompareConditionExecutorDoubleFloat;
import org.wso2.siddhi.core.executor.condition.compare.equal.EqualCompareConditionExecutorIntFloat;
import org.wso2.siddhi.core.executor.condition.compare.equal.EqualCompareConditionExpressionExecutorBoolBool;
import org.wso2.siddhi.core.executor.condition.compare.equal.EqualCompareConditionExpressionExecutorDoubleDouble;
import org.wso2.siddhi.core.executor.condition.compare.equal.EqualCompareConditionExpressionExecutorDoubleInt;
import org.wso2.siddhi.core.executor.condition.compare.equal.EqualCompareConditionExpressionExecutorDoubleLong;
import org.wso2.siddhi.core.executor.condition.compare.equal.EqualCompareConditionExpressionExecutorFloatDouble;
import org.wso2.siddhi.core.executor.condition.compare.equal.EqualCompareConditionExpressionExecutorFloatFloat;
import org.wso2.siddhi.core.executor.condition.compare.equal.EqualCompareConditionExpressionExecutorFloatInt;
import org.wso2.siddhi.core.executor.condition.compare.equal.EqualCompareConditionExpressionExecutorFloatLong;
import org.wso2.siddhi.core.executor.condition.compare.equal.EqualCompareConditionExpressionExecutorIntDouble;
import org.wso2.siddhi.core.executor.condition.compare.equal.EqualCompareConditionExpressionExecutorIntInt;
import org.wso2.siddhi.core.executor.condition.compare.equal.EqualCompareConditionExpressionExecutorIntLong;
import org.wso2.siddhi.core.executor.condition.compare.equal.EqualCompareConditionExpressionExecutorLongDouble;
import org.wso2.siddhi.core.executor.condition.compare.equal.EqualCompareConditionExpressionExecutorLongFloat;
import org.wso2.siddhi.core.executor.condition.compare.equal.EqualCompareConditionExpressionExecutorLongInt;
import org.wso2.siddhi.core.executor.condition.compare.equal.EqualCompareConditionExpressionExecutorLongLong;
import org.wso2.siddhi.core.executor.condition.compare.equal.EqualCompareConditionExpressionExecutorStringString;
import org.wso2.siddhi.core.executor.condition.compare.greater_than.GreaterThanCompareConditionExpressionExecutorDoubleDouble;
import org.wso2.siddhi.core.executor.condition.compare.greater_than.GreaterThanCompareConditionExpressionExecutorDoubleFloat;
import org.wso2.siddhi.core.executor.condition.compare.greater_than.GreaterThanCompareConditionExpressionExecutorDoubleInt;
import org.wso2.siddhi.core.executor.condition.compare.greater_than.GreaterThanCompareConditionExpressionExecutorDoubleLong;
import org.wso2.siddhi.core.executor.condition.compare.greater_than.GreaterThanCompareConditionExpressionExecutorFloatDouble;
import org.wso2.siddhi.core.executor.condition.compare.greater_than.GreaterThanCompareConditionExpressionExecutorFloatFloat;
import org.wso2.siddhi.core.executor.condition.compare.greater_than.GreaterThanCompareConditionExpressionExecutorFloatInt;
import org.wso2.siddhi.core.executor.condition.compare.greater_than.GreaterThanCompareConditionExpressionExecutorFloatLong;
import org.wso2.siddhi.core.executor.condition.compare.greater_than.GreaterThanCompareConditionExpressionExecutorIntDouble;
import org.wso2.siddhi.core.executor.condition.compare.greater_than.GreaterThanCompareConditionExpressionExecutorIntFloat;
import org.wso2.siddhi.core.executor.condition.compare.greater_than.GreaterThanCompareConditionExpressionExecutorIntInt;
import org.wso2.siddhi.core.executor.condition.compare.greater_than.GreaterThanCompareConditionExpressionExecutorIntLong;
import org.wso2.siddhi.core.executor.condition.compare.greater_than.GreaterThanCompareConditionExpressionExecutorLongDouble;
import org.wso2.siddhi.core.executor.condition.compare.greater_than.GreaterThanCompareConditionExpressionExecutorLongFloat;
import org.wso2.siddhi.core.executor.condition.compare.greater_than.GreaterThanCompareConditionExpressionExecutorLongInt;
import org.wso2.siddhi.core.executor.condition.compare.greater_than.GreaterThanCompareConditionExpressionExecutorLongLong;
import org.wso2.siddhi.core.executor.condition.compare.greater_than_equal.GreaterThanEqualCompareConditionExpressionExecutorDoubleDouble;
import org.wso2.siddhi.core.executor.condition.compare.greater_than_equal.GreaterThanEqualCompareConditionExpressionExecutorDoubleFloat;
import org.wso2.siddhi.core.executor.condition.compare.greater_than_equal.GreaterThanEqualCompareConditionExpressionExecutorDoubleInt;
import org.wso2.siddhi.core.executor.condition.compare.greater_than_equal.GreaterThanEqualCompareConditionExpressionExecutorDoubleLong;
import org.wso2.siddhi.core.executor.condition.compare.greater_than_equal.GreaterThanEqualCompareConditionExpressionExecutorFloatDouble;
import org.wso2.siddhi.core.executor.condition.compare.greater_than_equal.GreaterThanEqualCompareConditionExpressionExecutorFloatFloat;
import org.wso2.siddhi.core.executor.condition.compare.greater_than_equal.GreaterThanEqualCompareConditionExpressionExecutorFloatInt;
import org.wso2.siddhi.core.executor.condition.compare.greater_than_equal.GreaterThanEqualCompareConditionExpressionExecutorFloatLong;
import org.wso2.siddhi.core.executor.condition.compare.greater_than_equal.GreaterThanEqualCompareConditionExpressionExecutorIntDouble;
import org.wso2.siddhi.core.executor.condition.compare.greater_than_equal.GreaterThanEqualCompareConditionExpressionExecutorIntFloat;
import org.wso2.siddhi.core.executor.condition.compare.greater_than_equal.GreaterThanEqualCompareConditionExpressionExecutorIntInt;
import org.wso2.siddhi.core.executor.condition.compare.greater_than_equal.GreaterThanEqualCompareConditionExpressionExecutorIntLong;
import org.wso2.siddhi.core.executor.condition.compare.greater_than_equal.GreaterThanEqualCompareConditionExpressionExecutorLongDouble;
import org.wso2.siddhi.core.executor.condition.compare.greater_than_equal.GreaterThanEqualCompareConditionExpressionExecutorLongFloat;
import org.wso2.siddhi.core.executor.condition.compare.greater_than_equal.GreaterThanEqualCompareConditionExpressionExecutorLongInt;
import org.wso2.siddhi.core.executor.condition.compare.greater_than_equal.GreaterThanEqualCompareConditionExpressionExecutorLongLong;
import org.wso2.siddhi.core.executor.condition.compare.less_than.LessThanCompareConditionExpressionExecutorDoubleDouble;
import org.wso2.siddhi.core.executor.condition.compare.less_than.LessThanCompareConditionExpressionExecutorDoubleFloat;
import org.wso2.siddhi.core.executor.condition.compare.less_than.LessThanCompareConditionExpressionExecutorDoubleInt;
import org.wso2.siddhi.core.executor.condition.compare.less_than.LessThanCompareConditionExpressionExecutorDoubleLong;
import org.wso2.siddhi.core.executor.condition.compare.less_than.LessThanCompareConditionExpressionExecutorFloatDouble;
import org.wso2.siddhi.core.executor.condition.compare.less_than.LessThanCompareConditionExpressionExecutorFloatFloat;
import org.wso2.siddhi.core.executor.condition.compare.less_than.LessThanCompareConditionExpressionExecutorFloatInt;
import org.wso2.siddhi.core.executor.condition.compare.less_than.LessThanCompareConditionExpressionExecutorFloatLong;
import org.wso2.siddhi.core.executor.condition.compare.less_than.LessThanCompareConditionExpressionExecutorIntDouble;
import org.wso2.siddhi.core.executor.condition.compare.less_than.LessThanCompareConditionExpressionExecutorIntFloat;
import org.wso2.siddhi.core.executor.condition.compare.less_than.LessThanCompareConditionExpressionExecutorIntInt;
import org.wso2.siddhi.core.executor.condition.compare.less_than.LessThanCompareConditionExpressionExecutorIntLong;
import org.wso2.siddhi.core.executor.condition.compare.less_than.LessThanCompareConditionExpressionExecutorLongDouble;
import org.wso2.siddhi.core.executor.condition.compare.less_than.LessThanCompareConditionExpressionExecutorLongFloat;
import org.wso2.siddhi.core.executor.condition.compare.less_than.LessThanCompareConditionExpressionExecutorLongInt;
import org.wso2.siddhi.core.executor.condition.compare.less_than.LessThanCompareConditionExpressionExecutorLongLong;
import org.wso2.siddhi.core.executor.condition.compare.less_than_equal.LessThanEqualCompareConditionExpressionExecutorDoubleDouble;
import org.wso2.siddhi.core.executor.condition.compare.less_than_equal.LessThanEqualCompareConditionExpressionExecutorDoubleFloat;
import org.wso2.siddhi.core.executor.condition.compare.less_than_equal.LessThanEqualCompareConditionExpressionExecutorDoubleInt;
import org.wso2.siddhi.core.executor.condition.compare.less_than_equal.LessThanEqualCompareConditionExpressionExecutorDoubleLong;
import org.wso2.siddhi.core.executor.condition.compare.less_than_equal.LessThanEqualCompareConditionExpressionExecutorFloatDouble;
import org.wso2.siddhi.core.executor.condition.compare.less_than_equal.LessThanEqualCompareConditionExpressionExecutorFloatFloat;
import org.wso2.siddhi.core.executor.condition.compare.less_than_equal.LessThanEqualCompareConditionExpressionExecutorFloatInt;
import org.wso2.siddhi.core.executor.condition.compare.less_than_equal.LessThanEqualCompareConditionExpressionExecutorFloatLong;
import org.wso2.siddhi.core.executor.condition.compare.less_than_equal.LessThanEqualCompareConditionExpressionExecutorIntDouble;
import org.wso2.siddhi.core.executor.condition.compare.less_than_equal.LessThanEqualCompareConditionExpressionExecutorIntFloat;
import org.wso2.siddhi.core.executor.condition.compare.less_than_equal.LessThanEqualCompareConditionExpressionExecutorIntInt;
import org.wso2.siddhi.core.executor.condition.compare.less_than_equal.LessThanEqualCompareConditionExpressionExecutorIntLong;
import org.wso2.siddhi.core.executor.condition.compare.less_than_equal.LessThanEqualCompareConditionExpressionExecutorLongDouble;
import org.wso2.siddhi.core.executor.condition.compare.less_than_equal.LessThanEqualCompareConditionExpressionExecutorLongFloat;
import org.wso2.siddhi.core.executor.condition.compare.less_than_equal.LessThanEqualCompareConditionExpressionExecutorLongInt;
import org.wso2.siddhi.core.executor.condition.compare.less_than_equal.LessThanEqualCompareConditionExpressionExecutorLongLong;
import org.wso2.siddhi.core.executor.condition.compare.not_equal.NotEqualCompareConditionExpressionExecutorBoolBool;
import org.wso2.siddhi.core.executor.condition.compare.not_equal.NotEqualCompareConditionExpressionExecutorDoubleDouble;
import org.wso2.siddhi.core.executor.condition.compare.not_equal.NotEqualCompareConditionExpressionExecutorDoubleFloat;
import org.wso2.siddhi.core.executor.condition.compare.not_equal.NotEqualCompareConditionExpressionExecutorDoubleInt;
import org.wso2.siddhi.core.executor.condition.compare.not_equal.NotEqualCompareConditionExpressionExecutorDoubleLong;
import org.wso2.siddhi.core.executor.condition.compare.not_equal.NotEqualCompareConditionExpressionExecutorFloatDouble;
import org.wso2.siddhi.core.executor.condition.compare.not_equal.NotEqualCompareConditionExpressionExecutorFloatFloat;
import org.wso2.siddhi.core.executor.condition.compare.not_equal.NotEqualCompareConditionExpressionExecutorFloatInt;
import org.wso2.siddhi.core.executor.condition.compare.not_equal.NotEqualCompareConditionExpressionExecutorFloatLong;
import org.wso2.siddhi.core.executor.condition.compare.not_equal.NotEqualCompareConditionExpressionExecutorIntDouble;
import org.wso2.siddhi.core.executor.condition.compare.not_equal.NotEqualCompareConditionExpressionExecutorIntFloat;
import org.wso2.siddhi.core.executor.condition.compare.not_equal.NotEqualCompareConditionExpressionExecutorIntInt;
import org.wso2.siddhi.core.executor.condition.compare.not_equal.NotEqualCompareConditionExpressionExecutorIntLong;
import org.wso2.siddhi.core.executor.condition.compare.not_equal.NotEqualCompareConditionExpressionExecutorLongDouble;
import org.wso2.siddhi.core.executor.condition.compare.not_equal.NotEqualCompareConditionExpressionExecutorLongFloat;
import org.wso2.siddhi.core.executor.condition.compare.not_equal.NotEqualCompareConditionExpressionExecutorLongInt;
import org.wso2.siddhi.core.executor.condition.compare.not_equal.NotEqualCompareConditionExpressionExecutorLongLong;
import org.wso2.siddhi.core.executor.condition.compare.not_equal.NotEqualCompareConditionExpressionExecutorStringString;
import org.wso2.siddhi.core.executor.function.FunctionExecutor;
import org.wso2.siddhi.core.executor.math.Subtract.SubtractExpressionExecutorDouble;
import org.wso2.siddhi.core.executor.math.Subtract.SubtractExpressionExecutorFloat;
import org.wso2.siddhi.core.executor.math.Subtract.SubtractExpressionExecutorInt;
import org.wso2.siddhi.core.executor.math.Subtract.SubtractExpressionExecutorLong;
import org.wso2.siddhi.core.executor.math.add.AddExpressionExecutorDouble;
import org.wso2.siddhi.core.executor.math.add.AddExpressionExecutorFloat;
import org.wso2.siddhi.core.executor.math.add.AddExpressionExecutorInt;
import org.wso2.siddhi.core.executor.math.add.AddExpressionExecutorLong;
import org.wso2.siddhi.core.executor.math.divide.DivideExpressionExecutorDouble;
import org.wso2.siddhi.core.executor.math.divide.DivideExpressionExecutorFloat;
import org.wso2.siddhi.core.executor.math.divide.DivideExpressionExecutorInt;
import org.wso2.siddhi.core.executor.math.divide.DivideExpressionExecutorLong;
import org.wso2.siddhi.core.executor.math.mod.ModExpressionExecutorDouble;
import org.wso2.siddhi.core.executor.math.mod.ModExpressionExecutorFloat;
import org.wso2.siddhi.core.executor.math.mod.ModExpressionExecutorInt;
import org.wso2.siddhi.core.executor.math.mod.ModExpressionExecutorLong;
import org.wso2.siddhi.core.executor.math.multiply.MultiplyExpressionExecutorDouble;
import org.wso2.siddhi.core.executor.math.multiply.MultiplyExpressionExecutorFloat;
import org.wso2.siddhi.core.executor.math.multiply.MultiplyExpressionExecutorInt;
import org.wso2.siddhi.core.executor.math.multiply.MultiplyExpressionExecutorLong;
import org.wso2.siddhi.core.extension.holder.ExecutorExtensionHolder;
import org.wso2.siddhi.core.extension.holder.OutputAttributeExtensionHolder;
import org.wso2.siddhi.core.query.selector.attribute.handler.AttributeAggregator;
import org.wso2.siddhi.core.query.selector.attribute.processor.executor.AbstractAggregationAttributeExecutor;
import org.wso2.siddhi.core.query.selector.attribute.processor.executor.AggregationAttributeExecutor;
import org.wso2.siddhi.core.query.selector.attribute.processor.executor.GroupByAggregationAttributeExecutor;
import org.wso2.siddhi.core.util.SiddhiClassLoader;
import org.wso2.siddhi.query.api.definition.Attribute;
import org.wso2.siddhi.query.api.definition.StreamDefinition;
import org.wso2.siddhi.query.api.expression.Expression;
import org.wso2.siddhi.query.api.expression.Variable;
import org.wso2.siddhi.query.api.expression.condition.And;
import org.wso2.siddhi.query.api.expression.condition.Compare;
import org.wso2.siddhi.query.api.expression.condition.Not;
import org.wso2.siddhi.query.api.expression.condition.Or;
import org.wso2.siddhi.query.api.expression.constant.BoolConstant;
import org.wso2.siddhi.query.api.expression.constant.Constant;
import org.wso2.siddhi.query.api.expression.constant.DoubleConstant;
import org.wso2.siddhi.query.api.expression.constant.FloatConstant;
import org.wso2.siddhi.query.api.expression.constant.IntConstant;
import org.wso2.siddhi.query.api.expression.constant.LongConstant;
import org.wso2.siddhi.query.api.expression.constant.StringConstant;
import org.wso2.siddhi.query.api.expression.function.AttributeFunction;
import org.wso2.siddhi.query.api.expression.function.AttributeFunctionExtension;
import org.wso2.siddhi.query.api.expression.math.Add;
import org.wso2.siddhi.query.api.expression.math.Divide;
import org.wso2.siddhi.query.api.expression.math.Mod;
import org.wso2.siddhi.query.api.expression.math.Multiply;
import org.wso2.siddhi.query.api.expression.math.Subtract;
import org.wso2.siddhi.gpu.jni.SiddhiGpu;
import org.wso2.siddhi.gpu.jni.SiddhiGpu.ConstValue;
import org.wso2.siddhi.gpu.jni.SiddhiGpu.DataType;
import org.wso2.siddhi.gpu.jni.SiddhiGpu.ExecutorNode;
import org.wso2.siddhi.gpu.jni.SiddhiGpu.VariableValue;

public class GpuExpressionParser
{
	private static final Logger log = Logger.getLogger(GpuExpressionParser.class);

	public static SiddhiGpu.Filter parseExpression(Expression expression, SiddhiContext siddhiContext,
            ComplexMetaEvent metaEvent) {
		
		log.info("parseExpression");
		log.info("Root Expression = " + expression.toString());
		log.info("MetaEvent = " + metaEvent.toString());
		
		List<SiddhiGpu.ExecutorNode> gpuFilterList = new ArrayList<SiddhiGpu.ExecutorNode>();
		
		parseExpressionTree(expression, siddhiContext, metaEvent, gpuFilterList);
		
		SiddhiGpu.Filter filter = new SiddhiGpu.Filter(0, gpuFilterList.size());
		
		int i = 0;
		for (SiddhiGpu.ExecutorNode executorNode : gpuFilterList)
		{
			filter.AddExecutorNode(i, executorNode);
			i++;
		}
		
		return filter;
	}

	public static Attribute.Type parseExpressionTree(Expression expression, SiddhiContext siddhiContext,
			ComplexMetaEvent metaEvent,
			List<SiddhiGpu.ExecutorNode> gpuFilterList)
	{
		log.info("Expression = " + expression.toString());
		
		if (expression instanceof And)
		{
			gpuFilterList.add(new SiddhiGpu.ExecutorNode().SetNodeType(SiddhiGpu.EXECUTOR_NODE_CONDITION).SetConditionType(SiddhiGpu.EXECUTOR_AND));
			parseExpressionTree(((And) expression).getLeftExpression(), siddhiContext, metaEvent, gpuFilterList);
			parseExpressionTree(((And) expression).getRightExpression(), siddhiContext, metaEvent, gpuFilterList);
			return Attribute.Type.BOOL;
		}
		else if (expression instanceof Or)
		{
			gpuFilterList.add(new SiddhiGpu.ExecutorNode().SetNodeType(SiddhiGpu.EXECUTOR_NODE_CONDITION).SetConditionType(SiddhiGpu.EXECUTOR_OR));
			parseExpressionTree(((Or) expression).getLeftExpression(), siddhiContext, metaEvent, gpuFilterList);
			parseExpressionTree(((Or) expression).getRightExpression(), siddhiContext, metaEvent, gpuFilterList);
			return Attribute.Type.BOOL;
		}
		else if (expression instanceof Not)
		{
			gpuFilterList.add(new SiddhiGpu.ExecutorNode().SetNodeType(SiddhiGpu.EXECUTOR_NODE_CONDITION).SetConditionType(SiddhiGpu.EXECUTOR_NOT));
			parseExpressionTree(((Not) expression).getExpression(), siddhiContext, metaEvent, gpuFilterList);
			return Attribute.Type.BOOL;
		}
		else if (expression instanceof Compare)
		{
			if (((Compare) expression).getOperator() == Compare.Operator.EQUAL)
			{
				SiddhiGpu.ExecutorNode node = new SiddhiGpu.ExecutorNode().SetNodeType(SiddhiGpu.EXECUTOR_NODE_CONDITION);
				gpuFilterList.add(node);
				Attribute.Type lhs = parseExpressionTree(((Compare) expression).getLeftExpression(), siddhiContext, metaEvent, gpuFilterList);
				Attribute.Type rhs = parseExpressionTree(((Compare) expression).getRightExpression(), siddhiContext, metaEvent, gpuFilterList);
				parseEqualCompare(lhs, rhs, node);	
				return Attribute.Type.BOOL;
			}
			else if (((Compare) expression).getOperator() == Compare.Operator.NOT_EQUAL)
			{
				SiddhiGpu.ExecutorNode node = new SiddhiGpu.ExecutorNode().SetNodeType(SiddhiGpu.EXECUTOR_NODE_CONDITION);
				gpuFilterList.add(node);
				Attribute.Type lhs = parseExpressionTree(((Compare) expression).getLeftExpression(), siddhiContext, metaEvent, gpuFilterList);
				Attribute.Type rhs = parseExpressionTree(((Compare) expression).getRightExpression(), siddhiContext, metaEvent, gpuFilterList);
				parseNotEqualCompare(lhs, rhs, node);
				return Attribute.Type.BOOL;
			}
			else if (((Compare) expression).getOperator() == Compare.Operator.GREATER_THAN)
			{
				SiddhiGpu.ExecutorNode node = new SiddhiGpu.ExecutorNode().SetNodeType(SiddhiGpu.EXECUTOR_NODE_CONDITION);
				gpuFilterList.add(node);
				Attribute.Type lhs = parseExpressionTree(((Compare) expression).getLeftExpression(), siddhiContext, metaEvent, gpuFilterList);
				Attribute.Type rhs = parseExpressionTree(((Compare) expression).getRightExpression(), siddhiContext, metaEvent, gpuFilterList);
				parseGreaterThanCompare(lhs, rhs, node);
				return Attribute.Type.BOOL;
			}
			else if (((Compare) expression).getOperator() == Compare.Operator.GREATER_THAN_EQUAL)
			{
				SiddhiGpu.ExecutorNode node = new SiddhiGpu.ExecutorNode().SetNodeType(SiddhiGpu.EXECUTOR_NODE_CONDITION);
				gpuFilterList.add(node);
				Attribute.Type lhs = parseExpressionTree(((Compare) expression).getLeftExpression(), siddhiContext, metaEvent, gpuFilterList);
				Attribute.Type rhs = parseExpressionTree(((Compare) expression).getRightExpression(), siddhiContext, metaEvent, gpuFilterList);
				parseGreaterThanEqualCompare(lhs, rhs, node);
				return Attribute.Type.BOOL;
			}
			else if (((Compare) expression).getOperator() == Compare.Operator.LESS_THAN)
			{
				SiddhiGpu.ExecutorNode node = new SiddhiGpu.ExecutorNode().SetNodeType(SiddhiGpu.EXECUTOR_NODE_CONDITION);
				gpuFilterList.add(node);
				Attribute.Type lhs = parseExpressionTree(((Compare) expression).getLeftExpression(), siddhiContext, metaEvent, gpuFilterList);
				Attribute.Type rhs = parseExpressionTree(((Compare) expression).getRightExpression(), siddhiContext, metaEvent, gpuFilterList);
				parseLessThanCompare(lhs, rhs, node);
				return Attribute.Type.BOOL;
			}
			else if (((Compare) expression).getOperator() == Compare.Operator.LESS_THAN_EQUAL)
			{
				SiddhiGpu.ExecutorNode node = new SiddhiGpu.ExecutorNode().SetNodeType(SiddhiGpu.EXECUTOR_NODE_CONDITION);
				gpuFilterList.add(node);
				Attribute.Type lhs = parseExpressionTree(((Compare) expression).getLeftExpression(), siddhiContext, metaEvent, gpuFilterList);
				Attribute.Type rhs = parseExpressionTree(((Compare) expression).getRightExpression(), siddhiContext, metaEvent, gpuFilterList);
				parseLessThanEqualCompare(lhs, rhs, node);
				return Attribute.Type.BOOL;
			}
			else if (((Compare) expression).getOperator() == Compare.Operator.CONTAINS)
			{
				SiddhiGpu.ExecutorNode node = new SiddhiGpu.ExecutorNode().SetNodeType(SiddhiGpu.EXECUTOR_NODE_CONDITION);
				gpuFilterList.add(node);
				Attribute.Type lhs = parseExpressionTree(((Compare) expression).getLeftExpression(), siddhiContext, metaEvent, gpuFilterList);
				Attribute.Type rhs = parseExpressionTree(((Compare) expression).getRightExpression(), siddhiContext, metaEvent, gpuFilterList);
				parseContainsCompare(lhs, rhs, node);
				return Attribute.Type.BOOL;
			}

		}
		else if (expression instanceof Constant)
		{
			if (expression instanceof BoolConstant)
			{
				gpuFilterList.add(new ExecutorNode()
					.SetNodeType(SiddhiGpu.EXECUTOR_NODE_EXPRESSION)
					.SetExpressionType(SiddhiGpu.EXPRESSION_CONST)
					.SetConstValue(new ConstValue().SetBool(((BoolConstant) expression).getValue())));
				return Attribute.Type.BOOL;
			}
			else if (expression instanceof StringConstant)
			{
				String strVal = ((StringConstant) expression).getValue();
				gpuFilterList.add(new ExecutorNode()
					.SetNodeType(SiddhiGpu.EXECUTOR_NODE_EXPRESSION)
					.SetExpressionType(SiddhiGpu.EXPRESSION_CONST)
					.SetConstValue(new ConstValue().SetString(strVal, strVal.length())));
				return Attribute.Type.STRING;
			}
			else if (expression instanceof IntConstant)
			{
				gpuFilterList.add(new ExecutorNode()
					.SetNodeType(SiddhiGpu.EXECUTOR_NODE_EXPRESSION)
					.SetExpressionType(SiddhiGpu.EXPRESSION_CONST)
					.SetConstValue(new ConstValue().SetInt(((IntConstant) expression).getValue())));
				return Attribute.Type.INT;
			}
			else if (expression instanceof LongConstant)
			{
				gpuFilterList.add(new ExecutorNode()
					.SetNodeType(SiddhiGpu.EXECUTOR_NODE_EXPRESSION)
					.SetExpressionType(SiddhiGpu.EXPRESSION_CONST)
					.SetConstValue(new ConstValue().SetLong(((LongConstant) expression).getValue())));
				return Attribute.Type.LONG;
			}
			else if (expression instanceof FloatConstant)
			{
				gpuFilterList.add(new ExecutorNode()
					.SetNodeType(SiddhiGpu.EXECUTOR_NODE_EXPRESSION)
					.SetExpressionType(SiddhiGpu.EXPRESSION_CONST)
					.SetConstValue(new ConstValue().SetFloat(((FloatConstant) expression).getValue())));
				return Attribute.Type.FLOAT;
			}
			else if (expression instanceof DoubleConstant)
			{
				gpuFilterList.add(new ExecutorNode()
					.SetNodeType(SiddhiGpu.EXECUTOR_NODE_EXPRESSION)
					.SetExpressionType(SiddhiGpu.EXPRESSION_CONST)
					.SetConstValue(new ConstValue().SetDouble(((DoubleConstant) expression).getValue())));
				return Attribute.Type.DOUBLE;
			}

		}
		else if (expression instanceof Variable)
		{
			if (metaEvent instanceof MetaStreamEvent)
			{
				MetaStreamEvent metaStreamEvent = (MetaStreamEvent) metaEvent;
				Variable variable = (Variable) expression;
				String attributeName = variable.getAttributeName();
				
				StreamDefinition streamDef = (StreamDefinition) (metaStreamEvent.getInputDefinition());
				Attribute.Type type = streamDef.getAttributeType(attributeName);
				int position = streamDef.getAttributePosition(attributeName);
				
				int gpuDataType = SiddhiGpu.DataType.None;
				
				switch(type)
				{
					case BOOL:
						gpuDataType = SiddhiGpu.DataType.Boolean;
						break;
					case INT:
						gpuDataType = SiddhiGpu.DataType.Int;
						break;
					case LONG:
						gpuDataType = SiddhiGpu.DataType.Long;
						break;
					case FLOAT:
						gpuDataType = SiddhiGpu.DataType.Float;
						break;
					case DOUBLE:
						gpuDataType = SiddhiGpu.DataType.Double;
						break;
					case STRING:
						gpuDataType = SiddhiGpu.DataType.StringIn;
						break;
					case OBJECT:
						gpuDataType = SiddhiGpu.DataType.None;
						break;
				}
				
				gpuFilterList.add(new ExecutorNode()
					.SetNodeType(SiddhiGpu.EXECUTOR_NODE_EXPRESSION)
					.SetExpressionType(SiddhiGpu.EXPRESSION_VARIABLE)
					.SetVariableValue(new VariableValue(gpuDataType, position)));
				
				return type;
			}
			else
			{
				throw new OperationNotSupportedException("MetaStateEvents are not supported at the moment");
			}
			
		}
		else if (expression instanceof Multiply)
		{
			SiddhiGpu.ExecutorNode node = new SiddhiGpu.ExecutorNode().SetNodeType(SiddhiGpu.EXECUTOR_NODE_CONDITION);
			gpuFilterList.add(node);
			Attribute.Type lhs = parseExpressionTree(((Multiply) expression).getLeftValue(), siddhiContext, metaEvent, gpuFilterList);
			Attribute.Type rhs = parseExpressionTree(((Multiply) expression).getRightValue(), siddhiContext, metaEvent, gpuFilterList);
			Attribute.Type type = parseArithmeticOperationResultType(lhs, rhs);
			
			switch (type)
			{
				case INT:
					node.SetExpressionType(SiddhiGpu.EXPRESSION_MUL_INT);
					break;
				case LONG:
					node.SetExpressionType(SiddhiGpu.EXPRESSION_MUL_LONG);
					break;
				case FLOAT:
					node.SetExpressionType(SiddhiGpu.EXPRESSION_MUL_FLOAT);
					break;
				case DOUBLE:
					node.SetExpressionType(SiddhiGpu.EXPRESSION_MUL_DOUBLE);
					break;
				default: 
					break;
			}
			
			return type;
		}
		else if (expression instanceof Add)
		{
			SiddhiGpu.ExecutorNode node = new SiddhiGpu.ExecutorNode().SetNodeType(SiddhiGpu.EXECUTOR_NODE_CONDITION);
			gpuFilterList.add(node);
			Attribute.Type lhs = parseExpressionTree(((Add) expression).getLeftValue(), siddhiContext, metaEvent, gpuFilterList);
			Attribute.Type rhs = parseExpressionTree(((Add) expression).getRightValue(), siddhiContext, metaEvent, gpuFilterList);
			Attribute.Type type = parseArithmeticOperationResultType(lhs, rhs);
			
			switch (type)
			{
				case INT:
					node.SetExpressionType(SiddhiGpu.EXPRESSION_ADD_INT);
					break;
				case LONG:
					node.SetExpressionType(SiddhiGpu.EXPRESSION_ADD_LONG);
					break;
				case FLOAT:
					node.SetExpressionType(SiddhiGpu.EXPRESSION_ADD_FLOAT);
					break;
				case DOUBLE:
					node.SetExpressionType(SiddhiGpu.EXPRESSION_ADD_DOUBLE);
					break;
				default: 
					break;
			}
			
			return type;

		}
		else if (expression instanceof Subtract)
		{
			SiddhiGpu.ExecutorNode node = new SiddhiGpu.ExecutorNode().SetNodeType(SiddhiGpu.EXECUTOR_NODE_CONDITION);
			gpuFilterList.add(node);
			Attribute.Type lhs = parseExpressionTree(((Subtract) expression).getLeftValue(), siddhiContext, metaEvent, gpuFilterList);
			Attribute.Type rhs = parseExpressionTree(((Subtract) expression).getRightValue(), siddhiContext, metaEvent, gpuFilterList);
			Attribute.Type type = parseArithmeticOperationResultType(lhs, rhs);
			
			switch (type)
			{
				case INT:
					node.SetExpressionType(SiddhiGpu.EXPRESSION_SUB_INT);
					break;
				case LONG:
					node.SetExpressionType(SiddhiGpu.EXPRESSION_SUB_LONG);
					break;
				case FLOAT:
					node.SetExpressionType(SiddhiGpu.EXPRESSION_SUB_FLOAT);
					break;
				case DOUBLE:
					node.SetExpressionType(SiddhiGpu.EXPRESSION_SUB_DOUBLE);
					break;
				default: 
					break;
			}
			
			return type;
		}
		else if (expression instanceof Mod)
		{
			SiddhiGpu.ExecutorNode node = new SiddhiGpu.ExecutorNode().SetNodeType(SiddhiGpu.EXECUTOR_NODE_CONDITION);
			gpuFilterList.add(node);
			Attribute.Type lhs = parseExpressionTree(((Mod) expression).getLeftValue(), siddhiContext, metaEvent, gpuFilterList);
			Attribute.Type rhs = parseExpressionTree(((Mod) expression).getRightValue(), siddhiContext, metaEvent, gpuFilterList);
			Attribute.Type type = parseArithmeticOperationResultType(lhs, rhs);
			
			switch (type)
			{
				case INT:
					node.SetExpressionType(SiddhiGpu.EXPRESSION_MOD_INT);
					break;
				case LONG:
					node.SetExpressionType(SiddhiGpu.EXPRESSION_MOD_LONG);
					break;
				case FLOAT:
					node.SetExpressionType(SiddhiGpu.EXPRESSION_MOD_FLOAT);
					break;
				case DOUBLE:
					node.SetExpressionType(SiddhiGpu.EXPRESSION_MOD_DOUBLE);
					break;
				default: 
					break;
			}
			
			return type;
		}
		else if (expression instanceof Divide)
		{
			SiddhiGpu.ExecutorNode node = new SiddhiGpu.ExecutorNode().SetNodeType(SiddhiGpu.EXECUTOR_NODE_CONDITION);
			gpuFilterList.add(node);
			Attribute.Type lhs = parseExpressionTree(((Divide) expression).getLeftValue(), siddhiContext, metaEvent, gpuFilterList);
			Attribute.Type rhs = parseExpressionTree(((Divide) expression).getRightValue(), siddhiContext, metaEvent, gpuFilterList);
			Attribute.Type type = parseArithmeticOperationResultType(lhs, rhs);
			
			switch (type)
			{
				case INT:
					node.SetExpressionType(SiddhiGpu.EXPRESSION_DIV_INT);
					break;
				case LONG:
					node.SetExpressionType(SiddhiGpu.EXPRESSION_DIV_LONG);
					break;
				case FLOAT:
					node.SetExpressionType(SiddhiGpu.EXPRESSION_DIV_FLOAT);
					break;
				case DOUBLE:
					node.SetExpressionType(SiddhiGpu.EXPRESSION_DIV_DOUBLE);
					break;
				default: 
					break;
			}
			
			return type;

		}
		
		throw new UnsupportedOperationException(expression.toString() + " not supported!");

	}
	 
	private static void parseGreaterThanCompare(Attribute.Type lhsType, Attribute.Type rhsType, SiddhiGpu.ExecutorNode node)
	{
		switch (lhsType)
		{
			case STRING:
			{
				throw new OperationNotSupportedException("string cannot used in greater than comparisons");
			}
			case INT:
			{
				switch (rhsType)
				{
					case STRING:
						throw new OperationNotSupportedException("int cannot be compared with string");
					case INT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GT_INT_INT);
						break;
					case LONG:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GT_INT_LONG);
						break;
					case FLOAT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GT_INT_FLOAT);
						break;
					case DOUBLE:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GT_INT_DOUBLE);
						break;
					case BOOL:
						throw new OperationNotSupportedException("int cannot be compared with bool");
					default:
						throw new OperationNotSupportedException("int cannot be compared with " + rhsType);
				}
				break;
			}
			case LONG:
			{
				switch (rhsType)
				{
					case STRING:
						throw new OperationNotSupportedException("long cannot be compared with string");
					case INT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GT_LONG_INT);
						break;
					case LONG:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GT_LONG_LONG);
						break;
					case FLOAT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GT_LONG_FLOAT);
						break;
					case DOUBLE:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GT_LONG_DOUBLE);
						break;
					case BOOL:
						throw new OperationNotSupportedException("long cannot be compared with bool");
					default:
						throw new OperationNotSupportedException("long cannot be compared with " + rhsType);
				}
				break;
			}
			case FLOAT:
			{
				switch (rhsType)
				{
					case STRING:
						throw new OperationNotSupportedException("float cannot be compared with string");
					case INT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GT_FLOAT_INT);
						break;
					case LONG:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GT_FLOAT_LONG);
						break;
					case FLOAT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GT_FLOAT_FLOAT);
						break;
					case DOUBLE:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GT_FLOAT_DOUBLE);
						break;
					case BOOL:
						throw new OperationNotSupportedException("float cannot be compared with bool");
					default:
						throw new OperationNotSupportedException("float cannot be compared with " + rhsType);
				}
				break;
			}
			case DOUBLE:
			{
				switch (rhsType)
				{
					case STRING:
						throw new OperationNotSupportedException("double cannot be compared with string");
					case INT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GT_DOUBLE_INT);
						break;
					case LONG:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GT_DOUBLE_LONG);
						break;
					case FLOAT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GT_DOUBLE_FLOAT);
						break;
					case DOUBLE:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GT_DOUBLE_DOUBLE);
						break;
					case BOOL:
						throw new OperationNotSupportedException("double cannot be compared with bool");
					default:
						throw new OperationNotSupportedException("double cannot be compared with " + rhsType);
				}
				break;
			}
			case BOOL:
			{
				throw new OperationNotSupportedException("bool cannot used in greater than comparisons");
			}
			default:
			{
				throw new OperationNotSupportedException(lhsType + " cannot be used in greater than comparisons");
			}
		}
	}

	private static void parseGreaterThanEqualCompare(Attribute.Type lhsType, Attribute.Type rhsType, SiddhiGpu.ExecutorNode node)
	{
		switch (lhsType)
		{
			case STRING:
			{
				throw new OperationNotSupportedException("string cannot used in greater than equal comparisons");
			}
			case INT:
			{
				switch (rhsType)
				{
					case STRING:
						throw new OperationNotSupportedException("int cannot be compared with string");
					case INT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GE_INT_INT);
						break;
					case LONG:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GE_INT_LONG);
						break;
					case FLOAT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GE_INT_FLOAT);
						break;
					case DOUBLE:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GE_INT_DOUBLE);
						break;
					case BOOL:
						throw new OperationNotSupportedException("int cannot be compared with bool");
					default:
						throw new OperationNotSupportedException("int cannot be compared with " + rhsType);
				}
				break;
			}
			case LONG:
			{
				switch (rhsType)
				{
					case STRING:
						throw new OperationNotSupportedException("long cannot be compared with string");
					case INT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GE_LONG_INT);
						break;
					case LONG:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GE_LONG_LONG);
						break;
					case FLOAT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GE_LONG_FLOAT);
						break;
					case DOUBLE:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GE_LONG_DOUBLE);
						break;
					case BOOL:
						throw new OperationNotSupportedException("long cannot be compared with bool");
					default:
						throw new OperationNotSupportedException("long cannot be compared with " + rhsType);
				}
				break;
			}
			case FLOAT:
			{
				switch (rhsType)
				{
					case STRING:
						throw new OperationNotSupportedException("float cannot be compared with string");
					case INT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GE_FLOAT_INT);
						break;
					case LONG:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GE_FLOAT_LONG);
						break;
					case FLOAT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GE_FLOAT_FLOAT);
						break;
					case DOUBLE:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GE_FLOAT_DOUBLE);
						break;
					case BOOL:
						throw new OperationNotSupportedException("float cannot be compared with bool");
					default:
						throw new OperationNotSupportedException("float cannot be compared with " + rhsType);
				}
				break;
			}
			case DOUBLE:
			{
				switch (rhsType)
				{
					case STRING:
						throw new OperationNotSupportedException("double cannot be compared with string");
					case INT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GE_DOUBLE_INT);
						break;
					case LONG:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GE_DOUBLE_LONG);
						break;
					case FLOAT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GE_DOUBLE_FLOAT);
						break;
					case DOUBLE:
						node.SetConditionType(SiddhiGpu.EXECUTOR_GE_DOUBLE_DOUBLE);
						break;
					case BOOL:
						throw new OperationNotSupportedException("double cannot be compared with bool");
					default:
						throw new OperationNotSupportedException("double cannot be compared with " + rhsType);
				}
				break;
			}
			case BOOL:
			{
				throw new OperationNotSupportedException("bool cannot used in greater than equal comparisons");
			}
			default:
			{
				throw new OperationNotSupportedException(lhsType + " cannot be used in greater than comparisons");
			}
		}
	}

	private static void parseLessThanCompare(Attribute.Type lhsType, Attribute.Type rhsType, SiddhiGpu.ExecutorNode node)
	{
		switch (lhsType)
		{
			case STRING:
			{
				throw new OperationNotSupportedException("string cannot used in less than comparisons");
			}
			case INT:
			{
				switch (rhsType)
				{
					case STRING:
						throw new OperationNotSupportedException("int cannot be compared with string");
					case INT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LT_INT_INT);
						break;
					case LONG:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LT_INT_LONG);
						break;
					case FLOAT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LT_INT_FLOAT);
						break;
					case DOUBLE:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LT_INT_DOUBLE);
						break;
					case BOOL:
						throw new OperationNotSupportedException("int cannot be compared with bool");
					default:
						throw new OperationNotSupportedException("int cannot be compared with " + rhsType);
				}
				break;
			}
			case LONG:
			{
				switch (rhsType)
				{
					case STRING:
						throw new OperationNotSupportedException("long cannot be compared with string");
					case INT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LT_LONG_INT);
						break;
					case LONG:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LT_LONG_LONG);
						break;
					case FLOAT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LT_LONG_FLOAT);
						break;
					case DOUBLE:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LT_LONG_DOUBLE);
						break;
					case BOOL:
						throw new OperationNotSupportedException("long cannot be compared with bool");
					default:
						throw new OperationNotSupportedException("long cannot be compared with " + rhsType);
				}
				break;
			}
			case FLOAT:
			{
				switch (rhsType)
				{
					case STRING:
						throw new OperationNotSupportedException("float cannot be compared with string");
					case INT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LT_FLOAT_INT);
						break;
					case LONG:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LT_FLOAT_LONG);
						break;
					case FLOAT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LT_FLOAT_FLOAT);
						break;
					case DOUBLE:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LT_FLOAT_DOUBLE);
						break;
					case BOOL:
						throw new OperationNotSupportedException("float cannot be compared with bool");
					default:
						throw new OperationNotSupportedException("float cannot be compared with " + rhsType);
				}
				break;
			}
			case DOUBLE:
			{
				switch (rhsType)
				{
					case STRING:
						throw new OperationNotSupportedException("double cannot be compared with string");
					case INT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LT_DOUBLE_INT);
						break;
					case LONG:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LT_DOUBLE_LONG);
						break;
					case FLOAT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LT_DOUBLE_FLOAT);
						break;
					case DOUBLE:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LT_DOUBLE_DOUBLE);
						break;
					case BOOL:
						throw new OperationNotSupportedException("double cannot be compared with bool");
					default:
						throw new OperationNotSupportedException("double cannot be compared with " + rhsType);
				}
				break;
			}
			case BOOL:
			{
				throw new OperationNotSupportedException("bool cannot used in less than comparisons");
			}
			default:
			{
				throw new OperationNotSupportedException(lhsType + " cannot be used in less than comparisons");
			}
		}
	}

	private static void parseLessThanEqualCompare(Attribute.Type lhsType, Attribute.Type rhsType, SiddhiGpu.ExecutorNode node)
	{
		switch (lhsType)
		{
			case STRING:
			{
				throw new OperationNotSupportedException("string cannot used in less than equal comparisons");
			}
			case INT:
			{
				switch (rhsType)
				{
					case STRING:
						throw new OperationNotSupportedException("int cannot be compared with string");
					case INT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LE_INT_INT);
						break;
					case LONG:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LE_INT_LONG);
						break;
					case FLOAT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LE_INT_FLOAT);
						break;
					case DOUBLE:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LE_INT_DOUBLE);
						break;
					case BOOL:
						throw new OperationNotSupportedException("int cannot be compared with bool");
					default:
						throw new OperationNotSupportedException("int cannot be compared with " + rhsType);
				}
				break;
			}
			case LONG:
			{
				switch (rhsType)
				{
					case STRING:
						throw new OperationNotSupportedException("long cannot be compared with string");
					case INT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LE_LONG_INT);
						break;
					case LONG:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LE_LONG_LONG);
						break;
					case FLOAT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LE_LONG_FLOAT);
						break;
					case DOUBLE:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LE_LONG_DOUBLE);
						break;
					case BOOL:
						throw new OperationNotSupportedException("long cannot be compared with bool");
					default:
						throw new OperationNotSupportedException("long cannot be compared with " + rhsType);
				}
				break;
			}
			case FLOAT:
			{
				switch (rhsType)
				{
					case STRING:
						throw new OperationNotSupportedException("float cannot be compared with string");
					case INT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LE_FLOAT_INT);
						break;
					case LONG:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LE_FLOAT_LONG);
						break;
					case FLOAT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LE_FLOAT_FLOAT);
						break;
					case DOUBLE:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LE_FLOAT_DOUBLE);
						break;
					case BOOL:
						throw new OperationNotSupportedException("float cannot be compared with bool");
					default:
						throw new OperationNotSupportedException("float cannot be compared with " + rhsType);
				}
				break;
			}
			case DOUBLE:
			{
				switch (rhsType)
				{
					case STRING:
						throw new OperationNotSupportedException("double cannot be compared with string");
					case INT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LE_DOUBLE_INT);
						break;
					case LONG:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LE_DOUBLE_LONG);
						break;
					case FLOAT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LE_DOUBLE_FLOAT);
						break;
					case DOUBLE:
						node.SetConditionType(SiddhiGpu.EXECUTOR_LE_DOUBLE_DOUBLE);
						break;
					case BOOL:
						throw new OperationNotSupportedException("double cannot be compared with bool");
					default:
						throw new OperationNotSupportedException("double cannot be compared with " + rhsType);
				}
				break;
			}
			case BOOL:
			{
				throw new OperationNotSupportedException("bool cannot used in less than equal comparisons");
			}
			default:
			{
				throw new OperationNotSupportedException(lhsType + " cannot be used in less than comparisons");
			}
		}
	}

	private static void parseEqualCompare(Attribute.Type lhsType, Attribute.Type rhsType, SiddhiGpu.ExecutorNode node)
	{
		switch (lhsType)
		{
			case STRING:
			{
				switch (rhsType)
				{
					case STRING:
						node.SetConditionType(SiddhiGpu.EXECUTOR_EQ_STRING_STRING);
						break;
					default:
						throw new OperationNotSupportedException("sting cannot be compared with " + rhsType);
				}
				break;
			}
			case INT:
			{
				switch (rhsType)
				{
					case STRING:
						throw new OperationNotSupportedException("int cannot be compared with string");
					case INT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_EQ_INT_INT);
						break;
					case LONG:
						node.SetConditionType(SiddhiGpu.EXECUTOR_EQ_INT_LONG);
						break;
					case FLOAT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_EQ_INT_FLOAT);
						break;
					case DOUBLE:
						node.SetConditionType(SiddhiGpu.EXECUTOR_EQ_INT_DOUBLE);
						break;
					case BOOL:
						throw new OperationNotSupportedException("int cannot be compared with bool");
					default:
						throw new OperationNotSupportedException("int cannot be compared with " + rhsType);
				}
				break;
			}
			case LONG:
			{
				switch (rhsType)
				{
					case STRING:
						throw new OperationNotSupportedException("long cannot be compared with string");
					case INT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_EQ_LONG_INT);
						break;
					case LONG:
						node.SetConditionType(SiddhiGpu.EXECUTOR_EQ_LONG_LONG);
						break;
					case FLOAT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_EQ_LONG_FLOAT);
						break;
					case DOUBLE:
						node.SetConditionType(SiddhiGpu.EXECUTOR_EQ_LONG_DOUBLE);
						break;
					case BOOL:
						throw new OperationNotSupportedException("long cannot be compared with bool");
					default:
						throw new OperationNotSupportedException("long cannot be compared with " + rhsType);
				}
				break;
			}
			case FLOAT:
			{
				switch (rhsType)
				{
					case STRING:
						throw new OperationNotSupportedException("float cannot be compared with string");
					case INT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_EQ_FLOAT_INT);
						break;
					case LONG:
						node.SetConditionType(SiddhiGpu.EXECUTOR_EQ_FLOAT_LONG);
						break;
					case FLOAT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_EQ_FLOAT_FLOAT);
						break;
					case DOUBLE:
						node.SetConditionType(SiddhiGpu.EXECUTOR_EQ_FLOAT_DOUBLE);
						break;
					case BOOL:
						throw new OperationNotSupportedException("float cannot be compared with bool");
					default:
						throw new OperationNotSupportedException("float cannot be compared with " + rhsType);
				}
				break;
			}
			case DOUBLE:
			{
				switch (rhsType)
				{
					case STRING:
						throw new OperationNotSupportedException("double cannot be compared with string");
					case INT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_EQ_DOUBLE_INT);
						break;
					case LONG:
						node.SetConditionType(SiddhiGpu.EXECUTOR_EQ_DOUBLE_LONG);
						break;
					case FLOAT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_EQ_DOUBLE_FLOAT);
						break;
					case DOUBLE:
						node.SetConditionType(SiddhiGpu.EXECUTOR_EQ_DOUBLE_DOUBLE);
						break;
					case BOOL:
						throw new OperationNotSupportedException("double cannot be compared with bool");
					default:
						throw new OperationNotSupportedException("double cannot be compared with " + rhsType);
				}
				break;
			}
			case BOOL:
			{
				switch (rhsType)
				{
					case BOOL:
						node.SetConditionType(SiddhiGpu.EXECUTOR_EQ_BOOL_BOOL);
						break;
					default:
						throw new OperationNotSupportedException("bool cannot be compared with " + rhsType);
				}
				break;
			}
			default:
			{
				throw new OperationNotSupportedException(lhsType + " cannot be used in equal comparisons");
			}
		}
	}

	private static void parseNotEqualCompare(Attribute.Type lhsType, Attribute.Type rhsType, SiddhiGpu.ExecutorNode node)
	{
		switch (lhsType)
		{
			case STRING:
			{
				switch (rhsType)
				{
					case STRING:
						node.SetConditionType(SiddhiGpu.EXECUTOR_NE_STRING_STRING);
						break;
					default:
						throw new OperationNotSupportedException("sting cannot be compared with " + rhsType);
				}
				break;
			}
			case INT:
			{
				switch (rhsType)
				{
					case STRING:
						throw new OperationNotSupportedException("int cannot be compared with string");
					case INT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_NE_INT_INT);
						break;
					case LONG:
						node.SetConditionType(SiddhiGpu.EXECUTOR_NE_INT_LONG);
						break;
					case FLOAT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_NE_INT_FLOAT);
						break;
					case DOUBLE:
						node.SetConditionType(SiddhiGpu.EXECUTOR_NE_INT_DOUBLE);
						break;
					case BOOL:
						throw new OperationNotSupportedException("int cannot be compared with bool");
					default:
						throw new OperationNotSupportedException("int cannot be compared with " + rhsType);
				}
				break;
			}
			case LONG:
			{
				switch (rhsType)
				{
					case STRING:
						throw new OperationNotSupportedException("long cannot be compared with string");
					case INT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_NE_LONG_INT);
						break;
					case LONG:
						node.SetConditionType(SiddhiGpu.EXECUTOR_NE_LONG_LONG);
						break;
					case FLOAT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_NE_LONG_FLOAT);
						break;
					case DOUBLE:
						node.SetConditionType(SiddhiGpu.EXECUTOR_NE_LONG_DOUBLE);
						break;
					case BOOL:
						throw new OperationNotSupportedException("long cannot be compared with bool");
					default:
						throw new OperationNotSupportedException("long cannot be compared with " + rhsType);
				}
				break;
			}
			case FLOAT:
			{
				switch (rhsType)
				{
					case STRING:
						throw new OperationNotSupportedException("float cannot be compared with string");
					case INT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_NE_FLOAT_INT);
						break;
					case LONG:
						node.SetConditionType(SiddhiGpu.EXECUTOR_NE_FLOAT_LONG);
						break;
					case FLOAT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_NE_FLOAT_FLOAT);
						break;
					case DOUBLE:
						node.SetConditionType(SiddhiGpu.EXECUTOR_NE_FLOAT_DOUBLE);
						break;
					case BOOL:
						throw new OperationNotSupportedException("float cannot be compared with bool");
					default:
						throw new OperationNotSupportedException("float cannot be compared with " + rhsType);
				}
				break;
			}
			case DOUBLE:
			{
				switch (rhsType)
				{
					case STRING:
						throw new OperationNotSupportedException("double cannot be compared with string");
					case INT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_NE_DOUBLE_INT);
						break;
					case LONG:
						node.SetConditionType(SiddhiGpu.EXECUTOR_NE_DOUBLE_LONG);
						break;
					case FLOAT:
						node.SetConditionType(SiddhiGpu.EXECUTOR_NE_DOUBLE_FLOAT);
						break;
					case DOUBLE:
						node.SetConditionType(SiddhiGpu.EXECUTOR_NE_DOUBLE_DOUBLE);
						break;
					case BOOL:
						throw new OperationNotSupportedException("double cannot be compared with bool");
					default:
						throw new OperationNotSupportedException("double cannot be compared with " + rhsType);
				}
				break;
			}
			case BOOL:
			{
				switch (rhsType)
				{
					case BOOL:
						node.SetConditionType(SiddhiGpu.EXECUTOR_NE_BOOL_BOOL);
						break;
					default:
						throw new OperationNotSupportedException("bool cannot be compared with " + rhsType);
				}
				break;
			}
			default:
			{
				throw new OperationNotSupportedException(lhsType + " cannot be used in not equal comparisons");
			}
		}
	}

	private static void parseContainsCompare(Attribute.Type lhsType, Attribute.Type rhsType, SiddhiGpu.ExecutorNode node)
	{
		switch (lhsType)
		{
			case STRING:
			{
				switch (rhsType)
				{
					case STRING:
						node.SetConditionType(SiddhiGpu.EXECUTOR_CONTAINS);
						break;
					default:
						throw new OperationNotSupportedException(rhsType + " cannot be used in contains comparisons");
				}
				break;
			}
			default:
			{
				throw new OperationNotSupportedException(lhsType + " cannot be used in contains comparisons");
			}
		}
	}

	private static Attribute.Type parseArithmeticOperationResultType(Attribute.Type lhsType, Attribute.Type rhsType)
	{
		if (lhsType == Attribute.Type.DOUBLE || rhsType == Attribute.Type.DOUBLE)
		{
			return Attribute.Type.DOUBLE;
		}
		else if (lhsType == Attribute.Type.FLOAT || rhsType == Attribute.Type.FLOAT)
		{
			return Attribute.Type.FLOAT;
		}
		else if (lhsType == Attribute.Type.LONG || rhsType == Attribute.Type.LONG)
		{
			return Attribute.Type.LONG;
		}
		else if (lhsType == Attribute.Type.INT || rhsType == Attribute.Type.INT)
		{
			return Attribute.Type.INT;
		}
		else
		{
			throw new ArithmeticException("Arithmetic operation between " + lhsType + " and "
					+ rhsType + " cannot be executed");
		}
	}
}
