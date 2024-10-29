import pandas as pd

def convert_coordinates(points_df, piece_size=4000):
    """
    Convert coordinates from the original large image to coordinates in the corresponding smaller image pieces.
    
    Parameters:
    points_df: pandas DataFrame with columns 'x' and 'y' containing coordinates in the original image
    piece_size: size of each smaller image piece (default 4000)
    
    Returns:
    DataFrame with original coordinates and their corresponding piece information
    """
    def get_piece_info(x, y):
        # Find which piece the point belongs to
        piece_x = (x // piece_size) * piece_size
        piece_y = (y // piece_size) * piece_size
        
        # Calculate the sequence number based on the pattern
        x_index = piece_x // piece_size
        y_index = piece_y // piece_size
        sequence_number = x_index * 8 + y_index + 1
        
        # Calculate the relative coordinates within the piece
        relative_x = x - piece_x
        relative_y = y - piece_y
        
        return pd.Series({
            'piece_coordinates': f'[{piece_x},{piece_y},{piece_size},{piece_size}]',
            'sequence_number': sequence_number,
            'relative_x': relative_x,
            'relative_y': relative_y,
            'filename': f'29-17-IIDC.svs_[{piece_x},{piece_y},{piece_size},{piece_size}]_{sequence_number}.png'
        })

    # Apply the conversion to each point
    result = points_df.apply(lambda row: get_piece_info(row['x'], row['y']), axis=1)
    
    # Combine with original coordinates
    return pd.concat([points_df, result], axis=1)

# Read the Excel file
input_file = 'original_data/gaze_labelled/raw/29-17-IIDC.xlsx'
try:
    # Read Excel file, assuming first two columns are x and y coordinates
    points_df = pd.read_excel(input_file)
    
    # Rename columns if they're not already named 'x' and 'y'
    points_df.columns = ['x', 'y']
    
    # Convert coordinates
    results = convert_coordinates(points_df)
    
    # Print results in a nicely formatted way
    print("\nResults:")
    print("=========")
    for index, row in results.iterrows():
        print(f"\nPoint {index + 1}:")
        print(f"Original coordinates: ({row['x']}, {row['y']})")
        print(f"Found in image file: {row['filename']}")
        print(f"Relative coordinates within piece: ({row['relative_x']}, {row['relative_y']})")

except FileNotFoundError:
    print(f"Error: Could not find file '{input_file}'")
except Exception as e:
    print(f"Error processing file: {str(e)}")
