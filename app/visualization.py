from typing import Dict, Any, List, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from fastapi.responses import Response
import io
import base64
from enum import Enum

class ChartType(str, Enum):
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    BOX = "box"

class SecurityVisualizer:
    """Handles generation of various visualizations for security data"""
    
    @staticmethod
    def create_plot(
        data: List[Dict[str, Any]],
        chart_type: ChartType,
        x: Optional[str] = None,
        y: Optional[str] = None,
        title: str = "Security Data Visualization",
        color: Optional[str] = None,
        group_by: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Create a plotly visualization from query results
        
        Args:
            data: List of dictionaries containing the data
            chart_type: Type of chart to create
            x: Column to use for x-axis
            y: Column to use for y-axis (for bar/line/scatter)
            title: Chart title
            color: Column to use for color encoding
            group_by: Column to group by (for box plots, etc.)
            **kwargs: Additional arguments specific to chart types
            
        Returns:
            Base64 encoded PNG image of the plot
        """
        if not data:
            return ""
            
        df = pd.DataFrame(data)
        
        try:
            if chart_type == ChartType.BAR:
                fig = px.bar(
                    df, 
                    x=x, 
                    y=y, 
                    title=title,
                    color=color,
                    **kwargs
                )
            elif chart_type == ChartType.LINE:
                fig = px.line(
                    df, 
                    x=x, 
                    y=y, 
                    title=title,
                    color=color,
                    **kwargs
                )
            elif chart_type == ChartType.PIE:
                fig = px.pie(
                    df,
                    names=x,
                    values=y,
                    title=title,
                    **kwargs
                )
            elif chart_type == ChartType.SCATTER:
                fig = px.scatter(
                    df,
                    x=x,
                    y=y,
                    title=title,
                    color=color,
                    **kwargs
                )
            elif chart_type == ChartType.HEATMAP:
                # For heatmap, we need to pivot the data
                pivot_data = df.pivot(
                    index=kwargs.get('index', df.columns[0]),
                    columns=kwargs.get('columns', df.columns[1]),
                    values=kwargs.get('values', df.columns[2])
                )
                fig = px.imshow(
                    pivot_data,
                    labels=dict(x=kwargs.get('x_label', 'X'), 
                              y=kwargs.get('y_label', 'Y'),
                              color=kwargs.get('color_scale', 'Value')),
                    title=title
                )
            elif chart_type == ChartType.BOX:
                fig = px.box(
                    df,
                    x=group_by,
                    y=y,
                    title=title,
                    color=color,
                    **kwargs
                )
            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")
            
            # Update layout for better readability
            fig.update_layout(
                title=title,
                xaxis_title=x,
                yaxis_title=y,
                legend_title=color,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Arial", size=12, color="#2c3e50")
            )
            
            # Convert to base64 encoded image
            img_bytes = fig.to_image(format="png")
            return base64.b64encode(img_bytes).decode('utf-8')
            
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            return ""
    
    @staticmethod
    def get_chart_types() -> List[Dict[str, str]]:
        """Get available chart types and their descriptions"""
        return [
            {"value": "bar", "label": "Bar Chart", "description": "Compare values across categories"},
            {"value": "line", "label": "Line Chart", "description": "Show trends over time or ordered categories"},
            {"value": "pie", "label": "Pie Chart", "description": "Show proportions of a whole"},
            {"value": "scatter", "label": "Scatter Plot", "description": "Show relationships between two variables"},
            {"value": "heatmap", "label": "Heatmap", "description": "Show matrix data as colors"},
            {"value": "box", "label": "Box Plot", "description": "Show distribution and outliers"}
        ]

# Example usage:
if __name__ == "__main__":
    # Example data
    sample_data = [
        {"severity": "HIGH", "count": 15, "date": "2023-01-01"},
        {"severity": "MEDIUM", "count": 30, "date": "2023-01-01"},
        {"severity": "LOW", "count": 55, "date": "2023-01-01"},
        {"severity": "HIGH", "count": 10, "date": "2023-01-02"},
        {"severity": "MEDIUM", "count": 25, "date": "2023-01-02"},
        {"severity": "LOW", "count": 45, "date": "2023-01-02"},
    ]
    
    # Create a bar chart
    viz = SecurityVisualizer()
    img_data = viz.create_plot(
        sample_data,
        ChartType.BAR,
        x="date",
        y="count",
        color="severity",
        title="Alerts by Severity Over Time",
        barmode="group"
    )
    
    # Save to file for testing
    with open("sample_chart.png", "wb") as f:
        f.write(base64.b64decode(img_data))
