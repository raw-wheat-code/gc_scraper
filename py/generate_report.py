import pandas as pd
from fpdf import FPDF, XPos, YPos
from fpdf.fonts import FontFace
import os
import sys
import re

# Paths
HITTING_CSV = r"E:\GC Scraper\output\compiled_hitting_box_scores.csv"
PITCHING_CSV = r"E:\GC Scraper\output\compiled_pitching_stats.csv"
SCOUTING_CSV = r"E:\GC Scraper\output\compiled_scouting_metrics.csv"
POSITIONS_CSV = r"E:\GC Scraper\output\compiled_positions.csv"
CHARTS_DIR = r"E:\GC Scraper\charts"
REPORTS_DIR = r"E:\GC Scraper\reports"

class PlayerReport(FPDF):
    def __init__(self, team_name=""):
        super().__init__(orientation='L', unit='mm', format='A4')
        self.team_name = team_name
        self.alias_nb_pages()

    def header(self):
        self.set_font('Helvetica', 'B', 12)
        if self.team_name:
            self.cell(0, 10, self.team_name, new_x=XPos.LMARGIN, new_y=YPos.TOP, align='C')
        else:
            self.cell(0, 10, '', new_x=XPos.LMARGIN, new_y=YPos.TOP, align='C')
            
        self.set_font('Helvetica', '', 9)
        self.set_xy(250, 10)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align='R', new_x=XPos.LMARGIN, new_y=YPos.NEXT)

def clean_pdf_text(text):
    """
    Safely sanitizes text by swapping curly/smart quotes and other 
    non-Latin-1 characters into basic standard counterparts.
    """
    if pd.isna(text):
        return ""
    text = str(text)
    # Convert curly quotes and smart apostrophes
    text = text.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
    return text

def get_gradient_color(value, min_val, max_val, lower_is_better=False):
    """
    Computes a soft Red-Yellow-Green RGB color gradient based on a value's position.
    """
    if pd.isna(value) or max_val == min_val:
        return (255, 255, 200)
    
    t = (value - min_val) / (max_val - min_val)
    t = max(0.0, min(1.0, t))
    
    if lower_is_better:
        t = 1.0 - t
    
    if t < 0.5:
        factor = t / 0.5
        r = 255
        g = int(150 + (255 - 150) * factor)
        b = int(150 + (180 - 150) * factor)
    else:
        factor = (t - 0.5) / 0.5
        r = int(255 - (255 - 150) * factor)
        g = 255
        b = int(180 - (180 - 150) * factor)
        
    return (r, g, b)

def clean_filename_from_team(team_name):
    """
    Converts text to lowercase, removes special characters, and separates blocks with underscores.
    Example: "Lady Mustangs 14U - Harpold (Fall)" -> "lady_mustangs_14u_harpold_fall.pdf"
    """
    if not team_name:
        return "statistical_report.pdf"
    
    s = team_name.lower()
    s = re.sub(r'[^a-z0-9\s\-_]', '', s)
    s = re.sub(r'[\s\-_]+', ' ', s)
    s = s.strip().replace(' ', '_')
    
    return f"{s}.pdf"

def generate_report(team_name=""):
    df_hit = pd.read_csv(HITTING_CSV)
    df_pitch = pd.read_csv(PITCHING_CSV)
    df_scout = pd.read_csv(SCOUTING_CSV)
    
    active_pitchers = df_pitch[df_pitch['TP'] > 0].copy()
    
    full_df = pd.merge(df_hit, df_pitch[['Player Name', 'IP']], on='Player Name', how='left')
    full_df['IP'] = full_df['IP'].fillna(0)
    
    batting_summary_df = full_df.sort_values(by=['G', 'AVG', 'IP'], ascending=[False, False, False])
    pitching_summary_df = active_pitchers.sort_values(by=['ERA', 'WHIP', 'SO (P)'], ascending=[True, True, False])
    players_order = batting_summary_df['Player Name'].unique()
    
    bold_batting_metrics = ['3B', 'HR', 'BB']
    batting_leaders = {metric: batting_summary_df[metric].max() for metric in bold_batting_metrics}
    
    batting_color_metrics = ['H', 'SO', 'AVG', 'OBP', 'SLG', 'OPS']
    batting_bounds = {col: {'min': batting_summary_df[col].min(), 'max': batting_summary_df[col].max()} for col in batting_color_metrics}
    
    pitching_color_metrics = ['H (P)', 'SO (P)', 'WP', 'ERA', 'WHIP', 'Strike %']
    pitching_bounds = {col: {'min': active_pitchers[col].min(), 'max': active_pitchers[col].max()} for col in pitching_color_metrics}

    pdf = PlayerReport(team_name=team_name)
    
    # --- PAGE 1: MASTER BATTING SUMMARY TABLE (LANDSCAPE) ---
    pdf.add_page(orientation='L')
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, "Team Batting Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(2)
    
    summary_cols = ['Player Name', 'G', 'PA', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'AVG', 'OBP', 'SLG', 'OPS']
    col_widths_batting = (56, 14, 14, 14, 14, 14, 14, 14, 14, 14, 17, 17, 17, 17)
    
    pdf.set_font('Helvetica', '', 8)
    with pdf.table(line_height=6, text_align='CENTER', col_widths=col_widths_batting) as table:
        headers = table.row()
        for col in summary_cols:
            headers.cell(col)
            
        for _, row_data in batting_summary_df.iterrows():
            row = table.row()
            for col in summary_cols:
                val = row_data[col]
                
                is_bold = col in bold_batting_metrics and float(val) == batting_leaders[col]
                font_style = 'BOLD' if is_bold else ''
                
                if col in batting_color_metrics:
                    float_val = float(val)
                    cell_text = f"{float_val:.3f}" if col in ['AVG', 'OBP', 'SLG', 'OPS'] else str(int(float_val))
                    low_better = (col == 'SO')
                    bg_color = get_gradient_color(float_val, batting_bounds[col]['min'], batting_bounds[col]['max'], lower_is_better=low_better)
                    row.cell(cell_text, style=FontFace(emphasis=font_style, fill_color=bg_color))
                else:
                    if col == 'Player Name':
                        row.cell(clean_pdf_text(val))
                    elif col in ['G', 'PA', 'AB', '2B', '3B', 'HR', 'BB']:
                        row.cell(str(int(float(val))), style=FontFace(emphasis=font_style))
                    else:
                        row.cell(clean_pdf_text(val))

    # --- PAGE 2: MASTER PITCHING SUMMARY TABLE (LANDSCAPE) ---
    pdf.add_page(orientation='L')
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, "Team Pitching Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(2)
    
    pitching_cols = ['Player Name', 'IP', 'H (P)', 'R (P)', 'ER', 'BB (P)', 'SO (P)', 'WP', 'HBP (P)', 'TP', 'TS', 'ERA', 'WHIP', 'Strike %']
    col_widths_pitching = (41, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 19, 19, 19)
    
    pdf.set_font('Helvetica', '', 8)
    with pdf.table(line_height=6, text_align='CENTER', col_widths=col_widths_pitching) as table:
        headers = table.row()
        for col in pitching_cols:
            headers.cell(col)
            
        for _, row_data in pitching_summary_df.iterrows():
            row = table.row()
            for col in pitching_cols:
                val = row_data[col]
                
                if col in pitching_color_metrics:
                    float_val = float(val)
                    
                    if col in ['ERA', 'WHIP']:
                        cell_text = f"{float_val:.3f}"
                        low_better = True
                    elif col == 'Strike %':
                        cell_text = f"{float_val:.1%}"
                        low_better = False
                    else: 
                        cell_text = str(int(float_val))
                        low_better = col in ['H (P)', 'WP']
                        
                    bg_color = get_gradient_color(float_val, pitching_bounds[col]['min'], pitching_bounds[col]['max'], lower_is_better=low_better)
                    row.cell(cell_text, style=FontFace(fill_color=bg_color))
                else:
                    if col == 'Player Name':
                        row.cell(clean_pdf_text(val))
                    elif col in ['R (P)', 'ER', 'BB (P)', 'TP', 'TS']:
                        row.cell(str(int(float(val))))
                    elif col == 'IP':
                        row.cell(clean_pdf_text(val))
                    else:
                        row.cell(clean_pdf_text(val))
                        
        # --- PAGE 3: Positions TABLE (LANDSCAPE) ---
    pdf.add_page(orientation='L')
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, "Team Positional Breakdown", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(2)

    position_columns = ['Player Name', 'P', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF']
    col_widths_positions = (56, 14, 14, 14, 14, 14, 14, 14, 14, 14)

    pdf.set_font('Helvetica', '', 8)
    with pdf.table(line_height=6, text_align='CENTER', col_widths=col_widths_positions) as table:
        headers = table.row()
        for col in position_columns:
            headers.cell(col)
            
        df_positions = pd.read_csv(POSITIONS_CSV)
        for _, row_data in df_positions.iterrows():
            row = table.row()
            for col in position_columns:
                val = row_data[col]
                if col == 'Player Name':
                    row.cell(clean_pdf_text(val))
                else:
                    # Convert to float safely, handling missing values
                    float_val = float(val) if pd.notna(val) else 0.0
                    
                    if float_val == 0.0:
                        # Apply gray text color (#808080 -> RGB: 128, 128, 128)
                        row.cell(
                            "0.0%", 
                            style=FontFace(color=(128, 128, 128))
                        )
                    else:
                        # Default text color for non-zero percentages
                        row.cell(f"{float_val:.1%}")

    # --- INDIVIDUAL PLAYER PROFILE SHEETS (LANDSCAPE) ---
    for player in players_order:
        pdf.add_page(orientation='L')
        pdf.set_font('Helvetica', 'B', 16)
        pdf.cell(0, 10, clean_pdf_text(player), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        chart_path = os.path.join(CHARTS_DIR, f"team_{player.replace(' ', '_')}.jpg")
        if os.path.exists(chart_path):
            pdf.image(chart_path, x=10, y=30, w=100)
            
        pdf.set_xy(10, 135)
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 8, "Batting Statistics", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font('Helvetica', '', 10)
        
        h = df_hit[df_hit['Player Name'] == player].iloc[0]
        pdf.cell(0, 6, f"G: {int(h['G'])} | PA: {int(h['PA'])} | AB: {int(h['AB'])} | H: {int(h['H'])} | 2B: {int(h['2B'])} | 3B: {int(h['3B'])} | HR: {int(h['HR'])}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(0, 6, f"BB: {int(h['BB'])} | SO: {int(h['SO'])} | AVG: {h['AVG']:.3f} | OBP: {h['OBP']:.3f} | SLG: {h['SLG']:.3f} | OPS: {h['OPS']:.3f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        if player in df_pitch['Player Name'].values:
            pdf.ln(2)
            pdf.set_font('Helvetica', 'B', 11)
            pdf.cell(0, 6, "Pitching Statistics", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font('Helvetica', '', 9)
            p = df_pitch[df_pitch['Player Name'] == player].iloc[0]
            pdf.cell(0, 5, f"IP: {p['IP']} | TP: {int(p['TP'])} | H: {int(p['H (P)'])} | SO: {int(p['SO (P)'])} | WP: {int(p['WP'])} | Strike %: {p['Strike %']:.1%}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 5, f"ERA: {p['ERA']:.3f} | WHIP: {p['WHIP']:.3f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_xy(150, 30)
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 10, "Tendencies", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        scout = df_scout[df_scout['Player Name'] == player]
        for cat in ['Contact', 'Outcome']:
            pdf.set_x(150)
            pdf.set_font('Helvetica', 'B', 10)
            pdf.cell(0, 8, clean_pdf_text(cat), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            for _, row_data in scout[scout['Category'] == cat].sort_values('Value', ascending=False).iterrows():
                pdf.set_x(155)
                pdf.set_font('Helvetica', '', 9)
                pdf.cell(0, 6, f"{clean_pdf_text(row_data['Metric'])}: {row_data['Value']:.1%}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                
    final_pdf_name = clean_filename_from_team(team_name)
    pdf.output(os.path.join(REPORTS_DIR, final_pdf_name))

if __name__ == "__main__":
    provided_team_name = ""
    if len(sys.argv) > 1:
        provided_team_name = sys.argv[1]
        
    generate_report(team_name=provided_team_name)