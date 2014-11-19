package org.wso2.siddhi.core.query;

import java.util.ArrayList;
import java.util.List;

import javax.swing.text.StyledEditorKit.BoldAction;

import org.wso2.siddhi.core.util.SiddhiConstants;
import org.wso2.siddhi.query.api.annotation.Annotation;
import org.wso2.siddhi.query.api.annotation.Element;
import org.wso2.siddhi.query.api.exception.DuplicateAnnotationException;
import org.wso2.siddhi.query.api.util.AnnotationHelper;

/**
 * Class to hold configuration values set by annotations in a Query.
 * Used to pass these values between parsers when parsing the Query. 
 * @author prabodha
 */
public class QueryAnnotations {
	private List<Annotation> annotationList = new ArrayList<Annotation>();

	public QueryAnnotations()
	{
	}
	
	public QueryAnnotations(List<Annotation> annotations)
	{
		addAnnotations(annotations);
	}
	
	public void  addAnnotation(Annotation annotation) {
		annotationList.add(annotation);
	}
	
	public void  addAnnotations(List<Annotation> annotations) {
		for (Annotation ann : annotations)
		{
			annotationList.add(ann);
		}
	}

	public List<Annotation> getAnnotations() {
		return annotationList;
	}
	
	public String GetAnnotationStringValue(String annotationName, String elementName) {
		String value = null;
		try
		{
			Element element;
			element = AnnotationHelper.getAnnotationElement(annotationName, elementName, annotationList);
			if (element != null) {
				value = element.getValue();
			}
		}
		catch (DuplicateAnnotationException e)
		{
		}
        return value;
	}
	
	public boolean GetAnnotationBooleanValue(String annotationName, String elementName) {
		Boolean value = false;
		try
		{
			Element element;
			element = AnnotationHelper.getAnnotationElement(annotationName, elementName, annotationList);
			if (element != null) {
				value = SiddhiConstants.TRUE.equalsIgnoreCase(element.getValue());
			}
		}
		catch (DuplicateAnnotationException e)
		{
		}
        return value;
	}
	
	public Integer GetAnnotationIntegerValue(String annotationName, String elementName) {
		Integer value = null;
		try
		{
			Element element;
			element = AnnotationHelper.getAnnotationElement(annotationName, elementName, annotationList);
			if (element != null) {
				value = Integer.parseInt(element.getValue());
			}
		}
		catch (DuplicateAnnotationException e)
		{
		}
        return value;
	}
}
